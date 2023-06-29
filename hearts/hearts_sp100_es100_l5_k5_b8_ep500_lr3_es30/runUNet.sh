#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J Prepare

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=10

## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=32GB

## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=24:00:00 

## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plggraphcnn-gpu-a100

## Specyfikacja partycji
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:8

## Plik ze standardowym wyjściem
#SBATCH --output="output-%A.out"

## Plik ze standardowym wyjściem błędów
#SBATCH --error="error-%A.err"

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

source env/bin/activate

##python3 run_preprocessing.py --config /net/pr1/plgrid/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_prep.json
##python3 modifyPlans.py --config /net/pr1/plgrid/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_modify.json

##python3  run_training_UNet.py --config /net/pr1/plgrid/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_$1.json
##python3  run_training_UNet.py --config /net/archive/groups/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_1.json
##python3  run_training_UNet.py --config /net/archive/groups/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_2.json
##python3  run_training_UNet.py --config /net/archive/groups/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_3.json
##python3  run_training_UNet.py --config /net/archive/groups/plggonwelo/fromScratch/DoseModel/nnUNet/configs/config_train_UNET_4.json

mkdir $1
cp runUNet.sh $1
cp master_thesis_final.py $1

start=`date +%s`

python3 master_thesis_final.py $1 > $1/training.txt

## sprawdzenie pesymistycznego czasu startu: sbatch --test-only runUNet.sh

end=`date +%s`
runtime=$(((end-start)/60))
echo "$runtime minutes"

cp output* error* $1
rm -R output* error*
