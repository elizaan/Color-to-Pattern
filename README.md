# HOW TO RUN THE ML WORKFLOW

```bash
git clone https://github.com/pengyu965/ChartDete
cd ChartDete

conda create -n ChartDete python=3.8
conda activate ChartDete
pip install -v -e .

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -U openmim
mim install mmcv-full

python main.py
```

# HOW TO RUN THE CHART TO PATTERN

You have to be on the base folder (colot-to-pattern).

```bash
python main.py
```

## Output
The output can be found in ```bash output\patterned_bars``` . All the extended (in large number) outputs are [here](https://drive.google.com/drive/folders/1r9Xx3bHZ-2gxUZbnAmXuFpGfj1CiOdms?usp=sharing)
