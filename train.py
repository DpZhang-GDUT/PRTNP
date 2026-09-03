import torch
import torch.nn.functional as F
import numpy as np
import time
import tqdm
import os
import configargparse
torch.set_default_dtype(torch.float32)
from utils import *
from save_and_plot_utils import *
from opt import *
from model.QTTModel import QTTModel
from model.CPModel import CPModel
from model.VMModel import TensorVM
from model.TTModel import TTModel
from model.TuckerModel import TuckerModel
from robustbench.data import load_cifar10, load_cifar100, load_imagenet
from robustbench.utils import clean_accuracy, load_model
import torchattacks
from torchmetrics.functional import peak_signal_noise_ratio as psnr, structural_similarity_index_measure as ssim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils as vutils
from wavelet_soft_threshold import *

import wandb
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "9"


torch.set_num_threads(20)
torch.set_num_interop_threads(20)

MODEL_CLASSES = {
    'QTT': QTTModel,
    'TT': TTModel,
    'CP': CPModel,
    'Tucker': TuckerModel,
    'VM': TensorVM,
}

import warnings
warnings.filterwarnings("ignore")

def compute_consistency_metrics(clean_sample, reconstructed_sample):
    clean_sample = clean_sample.detach().float()
    reconstructed_sample = reconstructed_sample.detach().float()

    if clean_sample.dim() == 3:
        clean_sample = clean_sample.unsqueeze(0)
    if reconstructed_sample.dim() == 3:
        reconstructed_sample = reconstructed_sample.unsqueeze(0)

    if clean_sample.shape[-2:] != reconstructed_sample.shape[-2:]:
        clean_sample = F.interpolate(
            clean_sample,
            size=reconstructed_sample.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    clean_sample = torch.clamp(clean_sample, 0.0, 1.0)
    reconstructed_sample = torch.clamp(reconstructed_sample, 0.0, 1.0)

    clean_psnr = psnr(clean_sample, reconstructed_sample).item()
    clean_ssim = ssim(clean_sample, reconstructed_sample).item()
    clean_nrmse = (
        torch.norm(clean_sample - reconstructed_sample, p='fro')
        / (torch.norm(clean_sample, p='fro') + 1e-8)
    ).item()

    return clean_psnr, clean_ssim, clean_nrmse

def train(target, noisy_target, sampled_indices_all, model, clf, args, attack=False):
    set_seed(args.seed)
    use_wandb = args.use_wandb
    wandb_limited_logging = args.wandb_limited_logging

    if args.device_type == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_wandb:
        local_dir = "wandb_local"
        #check if wandb_local exists
        if not os.path.exists(local_dir) and not args.only_local_wandb:
            os.makedirs(local_dir)

        if args.only_local_wandb:
            wandb.init(project=f'{args.exp_name}', dir = local_dir, mode='dryrun')
        else:
            wandb.init(project=f'{args.exp_name}')

        num_upsampling_steps = len(args.iterations_for_upsampling)
        args.num_upsampling_steps = num_upsampling_steps
        wandb.config.update(args)  # Log hyperparameters
        

    if args.use_wandb and model.compression_factor is not None:
        wandb.log({"Compression_factor": model.compression_factor})

    grid_size = [args.init_reso for i in range(args.dimensions)]
    iterations = args.num_iterations
    iterations_for_upsampling, iterations_until_next_upsampling = calculate_iterations_until_next_upsampling(args, iterations)

    print("### New grid size {}, Rank {}".format(grid_size, model.max_rank))


    if args.subset_to_train_on == 1.0:
        sampler = SimpleSamplerImplicit(args.dimensions, batch_size=min(args.max_batch_size, int(model.current_reso**args.dimensions)), max_value=model.current_reso-1)
    else:
        # make a grid with 1s if in sampled_indices and 0s if not
        sampler, procentage_of_sampled_indices = get_subset_sampler(args, sampled_indices_all, model, args.default_val_for_non_sampled, masked_avg_pooling = args.masked_avg_pooling)
    
    # Initialize optimizer
    if "mlp" in args.model:
        optimizer = torch.optim.Adam(model.get_optparam_groups(lr_init=args.lr, lr_init_mlp=args.lr_init_mlp))
    else:
        optimizer = torch.optim.Adam(model.get_optparam_groups(lr_init=args.lr))


    # Get iterations until next upsampling and use it to determine warmup steps
    if iterations_for_upsampling[0] > iterations:
        iterations_lr_warmup = iterations
    else:   
        iterations_lr_warmup = iterations_for_upsampling[0] 
    
    # Scheduler
    lr_gamma = calculate_gamma(args.lr, args.lr_decay_factor_until_next_upsampling, iterations_lr_warmup)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    warmup_steps = args.warmup_steps
    if iterations_for_upsampling[0] > warmup_steps and len(iterations_for_upsampling) > 1:
        warmup_steps = iterations_for_upsampling[0]//2
    lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, warmup_steps)

    # Save data while training
    all_params = 0
    best_recon = None
    best_loss = 1e10
    losses = []
    validation_losses = []
    figsize=(16,8)
    psnr_val = -1
    saved_images = []
    saved_images_iterations = []
    save_times = []
    time_start = time.time()
    psnr_arr = []
    compression_rates = []
    recon_targets = []
    downsampled_targets = []
    putt_downsampled_targets = []
    
    model.to(device)
    
    loop_obj = tqdm(range(iterations),disable= not args.use_tqdm)
    time_start = time.time()

    y_rec = []
    SSIM = []
    PSNR = []
    NRMSE = []
    
    rank_ite = 0
    target_norm = torch.norm(model.target, p='fro').item()
    aux = None
    for ite in loop_obj:
        

        if ite in iterations_for_upsampling or ite == iterations - 1:
            
                reconstructed_diff = model.get_image_reconstruction().to(device)
                reconstructed_diff = reconstructed_diff/ model.diff_scale
                recon_targets.append(reconstructed_diff)
                print(f"Reconstruction at iteration {ite}: {reconstructed_diff.shape}, {reconstructed_diff.dtype}")

                all_params += model.num_trainable_params

        if ite in iterations_for_upsampling:
            with torch.no_grad():

                if ite == iterations_for_upsampling[0]:
                    print(f"upsampling at iteration {ite}")
                    
                    model = get_TN_model(args, target, noisy_target)
                    model.to(device)
                    
                    putt_downsampled_targets.append(model.downsampled_target)


                ite_index = iterations_for_upsampling.index(ite)

                downsampled_target_last = model.downsample_target(factor = int(model.target.shape[1]/model.current_reso), grayscale = model.grayscale, dim = model.dimensions, masked_avg_pooling = model.masked_avg_pooling)

                # save before and after upsampling
                model.max_rank = args.ranks_for_upsampling[rank_ite]
                rank_ite += 1
                sampler, optimizer, scheduler, best_loss, lr_warmup_scheduler, procentage_of_sampled_indices = upsample_dim(args, model, figsize,
                                                                        saved_images, saved_images_iterations, save_times, time_start, psnr_arr, compression_rates, 
                                                                        ite, ite_index, iterations_until_next_upsampling[ite_index], sampled_indices_all = sampled_indices_all)
                # print(ite, optimizer.param_groups)
                #print(f"Creating new max_rank  {model.max_rank}")
            
                if len(recon_targets) > 0:
                    accumulated_reconstruction = upsample_and_sum_targets(recon_targets, model.current_reso).to(device)
                    putt_downsampled_targets.append(model.downsampled_target)
                    model.downsampled_target = model.downsampled_target-accumulated_reconstruction
                    downsampled_targets.append(model.downsampled_target)


                # auxiliary variable for TV
                if ite == iterations_for_upsampling[-1]:
                    if noisy_target is not None:
                        aux = noisy_target.detach().to(device)
                    else:
                        print("Warning: noisy_target is None. It will use target instead.")
                        aux = target.detach().to(device)
                    aux = nn.Parameter(aux)
                    Sparse = nn.Parameter(torch.zeros_like(aux)) # sparse term
                    optimizer.add_param_group({'params': [aux, Sparse]})  # Optional: set a different learning rate


                if args.use_wandb and model.compression_factor is not None:
                    psnr_val = mse2psnr(loss.item())
                    log_metrics_wandb(ite+1, psnr_val, compression_factor = model.compression_factor)
                
                # Warmup steps
                warmup_steps = args.warmup_steps
                if ite_index + 1 < len(iterations_until_next_upsampling) and len(iterations_until_next_upsampling) > 1: # max half of iteration between upsampling
                    warmup_steps = (iterations_until_next_upsampling[ite_index]+ iterations_until_next_upsampling[ite_index +1])//2
                warmup_steps = ite + warmup_steps

                grid_size = [model.current_reso for i in range(args.dimensions)]
        
        
        # evaluate
        if (ite > iterations_for_upsampling[-1]) and (not attack) and(ite % 10 == 0):
            x_rec = model.get_image_reconstruction().detach().to(device)
            # be careful about the einsum
            target = torch.einsum('hwc->chw', model.target).unsqueeze(0).to(device)
            x_rec = torch.einsum('hwc->chw', x_rec).unsqueeze(0)
            minval = x_rec.min()
            maxval = x_rec.max()
            x_rec = (x_rec-minval)/(maxval-minval+1e-8)

            SSIM.append(ssim(target, x_rec).item())
            PSNR.append(psnr(target, x_rec).item())
            nrmse = torch.norm(target-x_rec, p='fro')/target_norm
            NRMSE.append(nrmse.item())

            # resize data before prediction
            if args.data == 'imagenet':
                x_rec = F.interpolate(x_rec, size=(224, 224), mode='bilinear')
            elif args.data in ['cifar10', 'cifar100']:
                x_rec = F.interpolate(x_rec, size=(32, 32), mode='bilinear')
            y_rec.append(torch.argmax(clf(x_rec)).item())

        optimizer.zero_grad()
        
        batch_indices, batch_indicies_norm = sampler.next_batch()

        loss, reg_term = model(batch_indices.to(device), batch_indicies_norm.to(device), recon_targets, ite, iterations)

        reg_term = reg_term * model.regularization_weight 
        

        if (ite > iterations_for_upsampling[-1]):
            total_loss = loss #+ torch.norm(aux-noisy_target, p='fro') + model.regularization_weight * TV(aux, p=1)
        else:
            total_loss = loss + reg_term

        total_loss.backward()
        optimizer.step()

        if len(recon_targets) > 0:
            sum_reconstruction = upsample_and_sum_targets(recon_targets, model.current_reso).to(device)
            sum_reconstruction += model.current_image/ model.diff_scale
            ori_dimen_loss = model.loss_fn(sum_reconstruction, model.downsampled_target_temp)
            losses.append(ori_dimen_loss.item())
        else:
            first_reconstruction = model.current_image/ model.diff_scale
            first_dimen_loss = model.loss_fn(first_reconstruction, model.downsampled_target_temp)
            losses.append(first_dimen_loss.item())

        if ite <= warmup_steps:
            lr_warmup_scheduler.step()
        else:
            scheduler.step()
        #scheduler.step()
        
        if ite % args.log_every == 0 or ite-1 in iterations_for_upsampling: # if log_every steps or just after upsampling
            if ite < warmup_steps:
                curr_lr = lr_warmup_scheduler.get_last_lr()[0]
            else:
                curr_lr = scheduler.get_last_lr()[0]
            
            psnr_val = mse2psnr(loss) # just an idea of global psnr
        
            loop_obj.set_postfix({"Current LR": curr_lr, "Current reso": model.current_reso, "PSNR": psnr_val})
            if reg_term > 0:
                # set loss and update regularization term
                loop_obj.set_description(f"Loss: {loss}")
                # loop_obj.set_description(f"Loss: {loss}, Reg: {reg_term}")
            else:
                loop_obj.set_description(f"Loss: {loss}")

            # Log metrics to wandb
            if use_wandb and not wandb_limited_logging:
                log_metrics_wandb(ite+1, loss=loss, curr_lr=curr_lr, model=model, grid_size=grid_size, psnr_val=psnr_val)

        # Save data while training
        if ite % args.save_every == 0 or ite in iterations_for_upsampling:
            if args.calculate_psnr_while_training:
                with torch.no_grad():
                    save_data_while_training(args, model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, ite)
        if loss.item() < best_loss:
            best_loss = loss.item()
            if reg_term > 0:
                # set loss and update regularization term
                loop_obj.set_description(f"Loss: {loss.item()}")
                #loop_obj.set_description(f"Loss: {loss.item()}, Reg: {reg_term.item()}")
            else: 
                loop_obj.set_description(f"Loss: {loss.item()}")
    
    time_end = time.time()
    print("Training time: " + str(time_end - time_start))
    print("Total number of parameters: ", all_params)
    # log training time
    if use_wandb:
        training_time = time_end - time_start
        log_metrics_wandb(ite+1, training_time=training_time)

    # Take model off GPU for potential memory issues
    model.target = model.target.cpu()
    model.downsampled_target = model.downsampled_target.cpu()
    
    ### PSNR and reconstruction of object ###
    if model.model == "QTT" and len(model.shape_factors) > 25: # PyTorch cannot permute more than 25 dimensions tensors - have to use batched reconstruction
        if args.noise_std > 0 and args.noise_type is not None or args.subset_to_train_on < 1.0:
            psnr_val, best_recon = model.batched_qtt(compute_reconstruction= args.compute_reconstruction, target = target) #best_recon might be None if
        else:
            psnr_val, best_recon = model.batched_qtt(compute_reconstruction= args.compute_reconstruction) #best_recon might be None if
    else:
        best_recon = model.get_image_reconstruction()
        psnr_val = psnr(model.target, best_recon.detach().cpu())

    #print("Best PSNR: " + str(psnr_val))

    upsampled_targets = [
                F.interpolate(
                    target.permute(2, 0, 1).unsqueeze(0), 
                    size=(model.current_reso, model.current_reso), 
                    mode='bilinear'
                ).squeeze(0).permute(1, 2, 0) 
                for target in recon_targets #[:-1]
            ]
    best_recon = sum(upsampled_targets)

    t = torch.einsum('hwc->chw', model.target).unsqueeze(0).to(device)
    x_rec = torch.einsum('hwc->chw', best_recon).unsqueeze(0)
    x_rec = torch.clamp(x_rec, 0.0, 1.0)
    final_ssim = ssim(t, x_rec)
    final_psnr = psnr(t, x_rec)
    final_nrmse = torch.norm(t-x_rec, p='fro')/torch.norm(t, p='fro')
    print(f"\n final reconstruction - PSNR: {final_psnr:.2f}, SSIM: {final_ssim:.4f}, NRMSE: {final_nrmse:.4f}")

    
    save_results(model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, ite, best_recon, use_wandb, psnr_val, noisy_target, target, losses, args)
    
    x_rec = torch.einsum('hwc->chw', best_recon).unsqueeze(0)
    return x_rec, y_rec, PSNR, SSIM, NRMSE, losses, final_psnr, final_ssim, final_nrmse, all_params, recon_targets, putt_downsampled_targets, downsampled_targets

def upsample_and_sum_targets(recon_targets, target_reso):

    import torch.nn.functional as F
    
    result = None
    
    for target in recon_targets:
        tensor = target.permute(2, 0, 1).unsqueeze(0)  # [1, 3, current_reso, current_reso]
        
        upsampled = F.interpolate(tensor, size=(target_reso, target_reso), mode='bilinear')
        
        upsampled = upsampled.squeeze(0).permute(1, 2, 0)  # [target_reso, target_reso, 3]
        
        if result is None:
            result = upsampled
        else:
            result += upsampled
    
    return result

def get_TN_model(args, target, noisy_target):
    model_args = get_model_args(args, target, noisy_target, args.device_type)
    if args.model in MODEL_CLASSES:
        model = MODEL_CLASSES[args.model]( **model_args)
        if args.model == "VM" and args.dimensions != 3:
            raise NotImplementedError("VM only implemented for 3D")
    else:   
        raise NotImplementedError("Model not implemented")
    
    return model

def get_noisy_data(x_test_adv, args):
    use_wandb = args.use_wandb
    set_seed(args.seed)

    target = x_test_adv.squeeze().detach()
    noisy_target = None

    # Noise Experiments
    if args.noise_std > 0 and args.noise_type != "None":
        if noisy_target is None:
            noisy_target = make_noisy_target(target, args)
        else:
            noisy_target = make_noisy_target(noisy_target, args)
    else:
        noisy_target = None

    # Train on Subset
    if args.subset_to_train_on < 1.0:
        print("#### TRAINING ON INCOMPLETE DATA ####")
        # Clone target to noisy_target
        if noisy_target is None:
            noisy_target = target.clone()

        if args.dimensions == 2:
            all_samples = torch.tensor(np.array(np.meshgrid(np.arange(target.shape[0]), np.arange(target.shape[1])))).T.reshape(-1, 2)
        elif args.dimensions == 3:
            all_samples = torch.tensor(np.array(np.meshgrid(np.arange(target.shape[0]), np.arange(target.shape[1]), np.arange(target.shape[2])))).T.reshape(-1, 3)
        else:
            raise NotImplementedError("Dimensions not implemented")

        num_indices = len(all_samples)
        all_samples = all_samples[torch.randperm(num_indices)]

        if args.is_random_box_impainting :
            # Generate a random square within the image dimensions
            image_height, image_width = target.shape[:2]
            box_samples = int(num_indices * (1 - args.subset_to_train_on))
            square_size = int(np.sqrt(box_samples))
            square_size = min(square_size, target.shape[0], target.shape[1])

            # Random start coordinates for the square
            start_x = np.random.randint(0, target.shape[1] - square_size)
            start_y = np.random.randint(0, target.shape[0] - square_size)

            # Define the square region as the non-sampled indices
            non_sampled_indices_all = [[y, x] for y in range(start_y, start_y + square_size) for x in range(start_x, start_x + square_size)]

            # Convert all_samples and non_sampled_indices_all to set of tuples for set difference operation
            all_samples_set = set(map(tuple, all_samples.tolist()))
            non_sampled_set = set(map(tuple, non_sampled_indices_all))

            # Define the sampled indices as all indices minus the non-sampled indices
            sampled_indices_all = np.array(list(all_samples_set - non_sampled_set))

            # Set the values within the square to the default value for non-sampled data
            for idx in non_sampled_indices_all:
                if args.dimensions == 2:
                    noisy_target[idx[0], idx[1]] = args.default_val_for_non_sampled
                elif args.dimensions == 3:
                    noisy_target[idx[0], idx[1], :] = args.default_val_for_non_sampled

        else:
            sampled_indices_all = all_samples[:int(num_indices * args.subset_to_train_on)]
            non_sampled_indices_all = all_samples[int(num_indices * args.subset_to_train_on):]
            
            # Set missing data points
            for idx in non_sampled_indices_all:
                if args.dimensions == 2:
                    noisy_target[idx[0], idx[1]] = args.default_val_for_non_sampled
                elif args.dimensions == 3:
                    noisy_target[idx[0], idx[1], :] = args.default_val_for_non_sampled

        if args.plot_subsampled_target:
            plt.imshow(noisy_target)
            plt.axis('off')
            #plt.savefig("noisy_target.png", bbox_inches='tight', pad_inches=0)
            plt.show()
            

        percentage_of_sampled_indices = len(sampled_indices_all) / len(all_samples)
        print(f"Using {percentage_of_sampled_indices * 100}% of all indices in downsampled target")
    else:
        sampled_indices_all = None

    # Adjust learning rate based on noise/incoplete data
    if noisy_target is not None and args.factor_reduce_lr_based_on_noise != 0:
        lr = args.lr
        if args.subset_to_train_on < 1.0:
            lr = lr * args.factor_reduce_lr_based_on_noise ** (1-args.subset_to_train_on) # lower subset_to_train_on requires lower lr
        elif args.noise_type == "gaussian" or args.noise_type == "laplace":
            lr = lr * args.factor_reduce_lr_based_on_noise ** args.noise_std # more noise requires lower lr
        args.lr = lr
        # update lr in wandb
        if use_wandb:
            wandb.config.update(args,allow_val_change=True)
        print("New Learning Rate: ", args.lr)

    return target, noisy_target, sampled_indices_all

def save_results(model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, ite, best_recon, use_wandb, psnr_val, noisy_target, target, losses, args):
    if args.calculate_psnr_while_training: # plot last image - Only for local runs
        with torch.no_grad():
            save_data_while_training(args, model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, ite )

    # SSIM
    ssim_val = calculate_ssim(model.target, best_recon, args.payload, args.dimensions, patched = False) # no OOM when patched = True
    #print("SSIM: " + str(ssim_val))

    if args.save_learned_recon:
        save_reconstruction(args, best_recon)

    if use_wandb:
        log_metrics_wandb(ite+1, psnr_best=psnr_val, ssim_val=ssim_val)
    
    # Usage
    if use_wandb and best_recon is not None and args.save_training_images:
        save_image_to_wandb(model, best_recon, ite, save_locally = args.save_images_locally_wandb, PSNR=psnr_val, SSIM=ssim_val)  # Assuming 'ite' is already defined in your code
        # if subset sampling, save noisy target
        if args.subset_to_train_on < 1.0:
            save_image_to_wandb(model, noisy_target, ite, save_locally = args.save_images_locally_wandb, PSNR=psnr_val+10, SSIM=ssim_val)
        
    if args.save_learned_recon or args.plot_3d_local:
        best_recon_np = None

        if best_recon is not None:
            # Check if best_recon is a PyTorch tensor and if it's on GPU
            if torch.is_tensor(best_recon):
                if best_recon.is_cuda:
                    best_recon_np = best_recon.detach().cpu().numpy()
                else:
                    best_recon_np = best_recon.detach().numpy()
            elif isinstance(best_recon, np.ndarray):
                best_recon_np = best_recon  # best_recon is already a numpy array

        # Check if best_recon_np was successfully created or assigned
        if best_recon_np is not None:
            # Construct the file path
            file_path = f"{args.target}_{args.model}_{args.target}_{args.max_rank}__{args.max_rank}_{args.init_reso}_{args.end_reso}_{args.num_iterations}_{args.seed}.npy"

            # save three slices
            slice_ = int(best_recon_np.shape[0]/2)
            
            best_slice_axial = best_recon_np[slice_,:,:]
            best_slice_coronal = best_recon_np[:,slice_,:]
            best_slice_sagittal = best_recon_np[:,:,slice_]
            target_slice_axial = target[slice_,:,:]
            target_slice_coronal = target[:,slice_,:]
            target_slice_sagittal = target[:,:,slice_]
            targets = [target_slice_axial, target_slice_coronal, target_slice_sagittal]
            best_slices = [best_slice_axial, best_slice_coronal, best_slice_sagittal]

            if args.save_learned_recon:
                # Save slices as npy file
                np.save(file_path + "_axial", best_slice_axial)
                np.save(file_path + "_coronal", best_slice_coronal)
                np.save(file_path + "_sagittal", best_slice_sagittal)       

            if args.plot_3d_local: 
                plot3dslices(targets, best_slices, figsize=figsize, title = "Target and Reconstruction", cmap = "gray")


    # Finish wandb run
    if use_wandb:
        wandb.finish()
    

    if not use_wandb and args.show_end_results_locally:
        gt_list = calculate_and_log_psnr(model, best_recon, noisy_target, use_wandb, args)
        plot_loss_and_saved_images(use_wandb, losses, saved_images, saved_images_iterations, psnr_arr, figsize, save_times, gt_list)
        save_results_locally(args, saved_images, saved_images_iterations, save_times, psnr_arr, gt_list, best_recon, model)
        plot_psnrs(args, psnr_arr, figsize)

def log_gradients(model):
    grad_log = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
                    # Wandb does not accept NoneType, so we check if the gradient is not None
            grad_log[f"grad_{name}"] = wandb.Histogram(param.grad.cpu().numpy()) if param.grad is not None else 0

    wandb.log(grad_log)


def get_subset_sampler(args, sampled_indices_all, model, default_val_for_non_sampled = 0.0, masked_avg_pooling = False):
    if args.payload > 1:
        grid_shape = model.target.shape[:-1]
    else:
        grid_shape = model.target.shape

    sampled_indices_grid = torch.zeros(grid_shape) + default_val_for_non_sampled

    if args.dimensions == 2:
        sampled_indices_grid[sampled_indices_all[:,0], sampled_indices_all[:,1]] = 1
    elif args.dimensions == 3:
        sampled_indices_grid[sampled_indices_all[:,0], sampled_indices_all[:,1], sampled_indices_all[:,2]] = 1
    else:
        raise NotImplementedError("Dimensions not implemented")

    # do average pooling using a factor of model.current_reso to get tiles allowed to be trained on
    factor = int(model.target.shape[0]/model.current_reso) 
    sampled_indices_grid = downsample_with_avg_pooling(sampled_indices_grid, factor, args.dimensions, grayscale = True, device = None, masked=masked_avg_pooling)

    # All where sampled_indices_grid is greater equal to default_val_for_non_sampled
    sampled_indices = torch.nonzero(sampled_indices_grid != default_val_for_non_sampled).squeeze() # 
    
    procentage_of_sampled_indices = len(sampled_indices)/len(sampled_indices_grid.view(-1))
    print("Using This Procentage of all indices in downsampled target", procentage_of_sampled_indices) # get sampled_indices proportion to total number of indices

    sampler = SimpleSamplerSubset(args.dimensions, batch_size=min(args.max_batch_size, int(model.current_reso**args.dimensions)), max_value=model.current_reso-1, indices = sampled_indices)
    return sampler, procentage_of_sampled_indices


def log_metrics_wandb(step, psnr_val=None, ssim_val=None, loss=None, curr_lr=None, model=None, training_time=None, grid_size=None, compression_factor=None, psnr_best=None, val_loss=None):
    metrics = {}  # Dictionary to store the metrics to be logged

    # Populate the metrics dictionary based on provided arguments
    if psnr_val is not None:
        metrics["PSNR"] = psnr_val
    if ssim_val is not None:
        metrics["SSIM"] = ssim_val
    if loss is not None:
        metrics["Loss"] = loss.item()
        if grid_size is not None:
            metrics[f"Loss{grid_size[0]}"] = loss.item()
    if curr_lr is not None:
        metrics["Current LR"] = curr_lr
    if model is not None and hasattr(model, 'current_reso'):
        metrics["Current reso"] = model.current_reso
    if training_time is not None:
        metrics["Training_time"] = training_time
    if compression_factor is not None:
        metrics["Compression_factor"] = compression_factor
    if psnr_best is not None:
        metrics["PSNR_best"] = psnr_best

    if val_loss is not None:
        metrics["Val_loss"] = val_loss
    # Log the metrics to wandb
    if metrics:
        try: 
            wandb.log(metrics, step=step)
        except:
            print("Could not log to wandb")



@torch.no_grad()
def setup_optimizer(args, model, iteration_index):
    new_lr = args.lr_decay_factor ** (iteration_index + 1) * args.lr

    if "mlp" in args.model.lower():
        new_lr_mlp = args.lr_decay_factor ** (iteration_index + 1) * args.lr_init_mlp
        optimizer = torch.optim.Adam(model.get_optparam_groups(lr_init=new_lr, lr_init_mlp=new_lr_mlp))
    else:
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
        except: # TT model
            optimizer = torch.optim.Adam(model.get_optparam_groups(lr_init=new_lr))

    return optimizer, new_lr

def setup_scheduler(optimizer, args, iterations_until_next_upsampling):
    lr_gamma = calculate_gamma(args.lr, args.lr_decay_factor_until_next_upsampling, iterations_until_next_upsampling)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    best_loss = 1e10  # reset best loss

    warmup_steps = args.warmup_steps
    lr_warmup_scheduler = linear_warmup_lr_scheduler(optimizer, warmup_steps)

    return scheduler, best_loss, lr_warmup_scheduler

def setup_sampler(args, sampled_indices_all, model):
    dimensions = model.dimensions

    if args.subset_to_train_on == 1.0:
        sampler = SimpleSamplerImplicit(dimensions, batch_size=min(args.max_batch_size, int(model.current_reso**dimensions)), max_value=model.current_reso-1)
        procentage_of_sampled_indices = None
    else:
        sampler, procentage_of_sampled_indices = get_subset_sampler(args, sampled_indices_all, model, masked_avg_pooling=args.masked_avg_pooling)

    return sampler, procentage_of_sampled_indices

def print_info(model, new_lr, dimensions, new_max_rank):
    if new_max_rank is not None:
        print(f"### New rank {new_max_rank}, lr {new_lr}, new_compression_factor {model.compression_factor}, model_size {model.sz_compressed_gb}")
    else:
        grid_size = [model.current_reso for _ in range(dimensions)]
        #print(f"### New grid_size {grid_size}, lr {new_lr}, new_compression_factor {model.compression_factor}, model_size {model.sz_compressed_gb}")
        print(f"### New grid_size {grid_size}, lr {new_lr}, new_rank {model.max_rank}")



def upsample_dim(args, model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, compression_rates, iteration, iteration_index, iterations_until_next_upsampling=1000, sampled_indices_all=None, new_max_rank=None):
    """
    Function that handles both upsample common and upsample dim functionalities.

    Args:
        args: Various arguments needed for the process.
        model: The model being used.
        figsize: Figure size for any plots or images.
        saved_images: A list to store saved images.
        saved_images_iterations: Iterations at which images are saved.
        save_times: A list to store the times at which data is saved.
        time_start: The start time of the process.
        psnr_arr: An array to store PSNR values.
        compression_rates: A list to store compression rates.
        iteration: The current iteration of the process.
        iteration_index: The index of the current iteration.
        iterations_until_next_upsampling: Iterations until the next upsample. Defaults to 1000. # used for lr warmup and scheduler
        sampled_indices_all: All sampled indices. Defaults to None meaning use all. # used for subset sampling
        new_max_rank: The new maximum rank, applicable for rank upsample. Defaults to None.
    
    Returns:
        A tuple containing the sampler, optimizer, scheduler, best_loss, lr_warmup_scheduler, and percentage_of_sampled_indices.
    """
    model.upsample(iteration_index)

    optimizer, new_lr = setup_optimizer(args, model, iteration_index)
    scheduler, best_loss, lr_warmup_scheduler = setup_scheduler(optimizer, args, iterations_until_next_upsampling)
    sampler, percentage_of_sampled_indices = setup_sampler(args, sampled_indices_all, model)  # Not needed for rank upsample

    print_info(model, new_lr, model.dimensions, None)

    if not should_skip_saving(model, args):
        save_data_while_training(args, model, figsize, saved_images, saved_images_iterations, save_times, time_start, psnr_arr, iteration)

    compression_rates.append(model.compression_factor)

    return sampler, optimizer, scheduler, best_loss, lr_warmup_scheduler, percentage_of_sampled_indices



class Model_g(nn.Module):
    def __init__(self, model, clf, args):
        super(Model_g, self).__init__()

        self.args = args

        self.generator = model

        self.clf = clf
        self.clf.eval()

    def forward(self, x_test_adv):

        device = self.args.device

        x_orig = x_test_adv.to(device)

        x_hwc = torch.einsum('chw->hwc', x_orig.squeeze()).float()

        target, noisy_target, sampled_indices_all = get_noisy_data(x_hwc, self.args)
        model = get_TN_model(args, target, noisy_target)

        x_rec, y_rec, PSNR, SSIM, NRMSE, losses, \
        final_psnr, final_ssim, final_nrmse, all_params, \
        recon_targets, putt_downsampled_targets, downsampled_targets = train(
            target, noisy_target, sampled_indices_all,
            model, self.clf, self.args, attack=True
        )

        

        return x_rec
    
    def classify(self, x):
        if getattr(self.args, 'data', None) == 'imagenet':
            if x.shape[-2:] != (224, 224):
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        elif getattr(self.args, 'data', None) in ['cifar10', 'cifar100']:
            if x.shape[-2:] != (32, 32):
                x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        return self.clf(x)

# main
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args, unknown = parser.parse_known_args()

    config = args.config
    args = get_kwargs_dict(config_file=config)

    print(args.data,args.attack,args.ranks_for_upsampling)
    
    is_valid = check_validity_upsampling_steps(args.init_reso, args.end_reso, args.iterations_for_upsampling, args.num_iterations)
    if not is_valid:
        print(" ##### !!!! ##### Invalid upsampling iterations for {} to {} with {} iterations and {} num iterations".format(args.init_reso, args.end_reso, args.iterations_for_upsampling, args.num_iterations))
        raise NotImplementedError("Invalid upsampling iterations")
    elif args.noise_type == 'None' and args.noise_std != 0.0 or args.noise_type != 'None' and args.noise_std == 0.0:
        print(" ########## Skipping combination with noise_type: {} and noise_std: {}".format(args.noise_type, args.noise_std))
        raise NotImplementedError("Invalid noise type")

    if args.device_type == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device

    model_name = args.model
    if (args.model == 'TT') and (args.is_tensor_ring):
        model_name = 'TR'
    if (args.model == 'QTT') and (args.is_tensor_ring):
        model_name = 'QTR'

    if not os.path.exists(f'adv/{args.data}_{args.end_reso}/{args.attack}/'):
        os.makedirs(f'adv/{args.data}_{args.end_reso}/{args.attack}/')
    if not os.path.exists(f'out/{args.data}_{args.end_reso}/{model_name}-{args.attack}/'):
        os.makedirs(f'out/{args.data}_{args.end_reso}/{model_name}-{args.attack}/')
    if not os.path.exists(f'viz/{args.data}_{args.end_reso}/{model_name}-{args.attack}/'):
        os.makedirs(f'viz/{args.data}_{args.end_reso}/{model_name}-{args.attack}/')
    if not os.path.exists(f'metrics/{args.clf}/{args.data}_{args.end_reso}/{model_name}-{args.attack}/'):
        os.makedirs(f'metrics/{args.clf}/{args.data}_{args.end_reso}/{model_name}-{args.attack}/')

    if args.data == 'cifar10':
        x_test, y_test = load_cifar10(n_examples=args.subset_size)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        '''data = torch.load("cifar10_split/part_29.pt")
        x_test = data['x'].to(device)
        y_test = data['y'].to(device)'''

        # clf = load_model('Standard', dataset='cifar10').to(device)
        clf = load_model('Cui2023Decoupled_WRN-28-10', dataset='cifar10').to(device)

    elif args.data == 'cifar100':
        x_test, y_test = load_cifar100(n_examples=args.subset_size)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        clf = load_model('Cui2023Decoupled_WRN-28-10', dataset='cifar100').to(device)
    elif args.data == 'imagenet':
        x_test, y_test = load_imagenet(n_examples=args.subset_size, data_dir='/home/dongping/reconstruction/datasets/ImageNet/data')
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        clf = load_model('Standard_R50', dataset='imagenet').to(device)
    else:
        raise ValueError(f"Data {args.data} not found")

    #### Attack batches cifar10, load the available attacked imagenet
    if ('+' not in args.attack) and (args.attack != 'clean'): # if attack only clf
        attacktype = args.attack.split('-')
        if attacktype[0] == 'AA':
            if attacktype[1] == 'linf':
                attack = torchattacks.AutoAttack(clf, norm='Linf', eps=8/255)
            elif attacktype[1] == 'l2':
                attack = torchattacks.AutoAttack(clf, norm='L2', eps=0.5)
        elif args.attack in ['pgdeot']:
            if args.data == 'imagenet':
                eps = 4/255
            elif args.data in ['cifar10', 'cifar100']:
                eps = 8/255
            attack = torchattacks.EOTPGD(clf, eps=eps, alpha=2/255, steps=20, eot_iter=20)
        elif args.attack == 'clean':
            attack = lambda x, y: x
        else:
            raise ValueError(f"Unsupported attack type: {args.attack}")

        if args.data in ['cifar10', 'cifar100',]:
            '''testset = TensorDataset(x_test, y_test)
            testloader = DataLoader(testset, batch_size=64)
            x_test_adv = []
            for batch_x, batch_y in testloader:
                adv = attack(batch_x, batch_y)
                x_test_adv.append(adv)
            x_test_adv = torch.cat(x_test_adv, dim=0)'''
            if args.attack == 'AA-linf':
                attacked = torch.load('adv/cifar10_32/AA-linf/512_8.pth')

            x_test_adv = attacked['x_test_adv'].to(device)
            y_test = attacked['y_test'].to(device)
        elif args.data == 'imagenet':
            if args.attack == 'AA-linf':
                attacked = torch.load('/home/dongping/reconstruction/adv/imagenet_256/AA-linf/512_4.pth')

            x_test_adv = attacked['x_test_adv'].to(device)
            y_test = attacked['y_test'].to(device)
    else: 
        x_test_adv = x_test

    # resize data
    if args.data == 'imagenet':
        x_test_adv = F.interpolate(x_test_adv, size=(args.end_reso, args.end_reso), mode='bilinear')
    elif args.data in ['cifar10', 'cifar100']:
        x_test_adv = F.interpolate(x_test_adv, size=(args.end_reso, args.end_reso), mode='bilinear')

    x_rec = []
    x = []

    nrmselist = []
    psnrlist = []
    ssimlist = []
    predlist = []
    all_losses_list = []
    final_psnrlist = []
    final_ssimlist = []
    final_nrmselist = []
    clean_consistency_psnr_list = []
    clean_consistency_ssim_list = []
    clean_consistency_nrmse_list = []
    final_all_params = []
    recon_targets_list = []
    downsampled_targets_list = []
    putt_downsampled_targets_list = []
   

    for i, x_adv in enumerate(x_test_adv):

        adv = x_adv.squeeze()

        vutils.save_image(
            adv.detach().cpu(),
            os.path.join(f'adv/{args.data}_{args.end_reso}/{args.attack}/', f'{i}.png'),
            normalize=True, range=[0, 1]
        )
        
        adv = torch.einsum('chw->hwc', adv).float()
        
        if args.dimensions == 3:
            adv = adv.unsqueeze(0).repeat(16, 1, 1, 1)

        target, noisy_target, sampled_indices_all = get_noisy_data(adv, args)
        model = get_TN_model(args, target, noisy_target)
        
        x_clean, y_clean, PSNR, SSIM, NRMSE, losses, final_psnr, final_ssim, final_nrmse, all_params, recon_targets, putt_downsampled_targets, downsampled_targets = train(target, noisy_target, sampled_indices_all, model, clf, args)
        

        vutils.save_image(
            x_clean.squeeze(),
            os.path.join(f'viz/{args.data}_{args.end_reso}/{model_name}-{args.attack}/', f'input_rec_{i}.png'),
            normalize=True, range=[0, 1]
        )

        clean_sample = x_test[i].unsqueeze(0).to(x_clean.device)
        clean_psnr, clean_ssim, clean_nrmse = compute_consistency_metrics(clean_sample, x_clean)
        clean_consistency_psnr_list.append(clean_psnr)
        clean_consistency_ssim_list.append(clean_ssim)
        clean_consistency_nrmse_list.append(clean_nrmse)
        print(
            f"Sample {i} clean consistency - "
            f"PSNR: {clean_psnr:.2f}, SSIM: {clean_ssim:.4f}, NRMSE: {clean_nrmse:.4f}"
        )

        x_rec.append(x_clean)

        nrmselist.append(NRMSE)
        ssimlist.append(SSIM)
        psnrlist.append(PSNR)
        predlist.append([(p==y_test[i]).item() for p in y_clean])
        all_losses_list.append(losses)
        final_psnrlist.append(final_psnr.detach().cpu().numpy())
        final_ssimlist.append(final_ssim.detach().cpu().numpy())
        final_nrmselist.append(final_nrmse.detach().cpu().numpy())
        final_all_params.append(all_params)
        recon_targets_list.append(recon_targets)
        downsampled_targets_list.append(downsampled_targets)
        putt_downsampled_targets_list.append(putt_downsampled_targets)


    print(f"{args.subset_size} samples :""Average final PSNR:", f"{np.mean(final_psnrlist):.2f}"," SSIM:", f"{np.mean(final_ssimlist):.4f}", "NRMSE:", f"{np.mean(final_nrmselist):.4f}")
    print(
        f"{args.subset_size} samples : Average clean consistency - "
        f"PSNR: {np.mean(clean_consistency_psnr_list):.2f}, "
        f"SSIM: {np.mean(clean_consistency_ssim_list):.4f}, "
        f"NRMSE: {np.mean(clean_consistency_nrmse_list):.4f}"
    )
    print(f"Average final number of parameters: {np.mean(final_all_params)}")
    # save the metrics
    if model_name in ['TR', 'QTR']:
        with open(f'metrics/{args.clf}/{args.data}_{args.end_reso}/{model_name}-{args.attack}/{args.max_rank}_{args.channel_rank}_{args.noise_std}.pkl', 'wb') as file:
            pickle.dump([nrmselist, ssimlist, psnrlist, predlist, all_losses_list, final_psnrlist, final_ssimlist, final_nrmselist, clean_consistency_psnr_list, clean_consistency_ssim_list, clean_consistency_nrmse_list, recon_targets_list, putt_downsampled_targets_list, downsampled_targets_list], file)
    else:
        with open(f'metrics/{args.clf}/{args.data}_{args.end_reso}/{model_name}-{args.attack}/{args.max_rank}_{args.noise_std}.pkl', 'wb') as file:
            pickle.dump([nrmselist, ssimlist, psnrlist, predlist, all_losses_list, final_psnrlist, final_ssimlist, final_nrmselist, clean_consistency_psnr_list, clean_consistency_ssim_list, clean_consistency_nrmse_list, recon_targets_list, putt_downsampled_targets_list, downsampled_targets_list], file)

    x_rec = torch.cat(x_rec, dim=0).to(device)

    # resize results to original size
    if args.data == 'imagenet':
        x_rec = F.interpolate(x_rec, size=(224, 224), mode='bilinear')
    elif args.data in ['cifar10', 'cifar100']:
        x_rec = F.interpolate(x_rec, size=(32, 32), mode='bilinear')

    print(f"{args.data} {args.attack} accuracy", clean_accuracy(clf, x_rec, y_test)) 