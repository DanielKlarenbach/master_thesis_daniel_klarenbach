#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J Prepare

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1

## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=40GB

## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=24:00:00 

## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plggraphcnn-gpu-a100

## Specyfikacja partycji
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1

## Plik ze standardowym wyjściem
#SBATCH --output="output-%A.out"

## Plik ze standardowym wyjściem błędów
#SBATCH --error="error-%A.err"

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

source $HOME/env/bin/activate

start=`date +%s`

python3 u.py $SLURM_SUBMIT_DIR > $SLURM_SUBMIT_DIR/training.txt


end=`date +%s`
runtime=$(((end-start)/60))
echo "$runtime minutes"
echo "Job id: $SLURM_JOB_ID" >> $SLURM_SUBMIT_DIR/training.txt
