from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os, multiprocessing as mp, torch.multiprocessing as tmp

def main():
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # trainer = 'SwinUNetTrainer__nnUNetPlans__2d'
    trainer = 'nnUNetTrainer__nnUNetPlans__2d'
    trainer = 'prev_meddinov3_base_Simple_Mask2Former_Trainer__nnUNetPlans__2d_896'
    dataset = 'Dataset515_CardiacSeg'
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, f'{dataset}/{trainer}'),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(join(nnUNet_raw, 'Dataset501_Total/imagesTs'),
                                    join(nnUNet_raw, f'Dataset501_Total/imagesTs_{trainer}'),
                                    save_probabilities=False, overwrite=False,
                                    num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    
if __name__ == "__main__":
    # safer defaults on HPC
    try:
        mp.set_start_method("fork")  # Linux: avoid spawn’s import requirements
    except RuntimeError:
        pass
   # Avoid /dev/shm
    tmp.set_sharing_strategy("file_system")

    # Keep CPU thread counts sane
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    main()
