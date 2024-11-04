successful rollouts examples:

### examples

unsuccessfull rollouts:

https://github.com/user-attachments/assets/21ed72bc-993b-4f9c-a3be-96eb8a9b64b7

https://github.com/user-attachments/assets/5f54ec5f-57a7-4262-9af6-918b39c49f35

unsuccessfull rollouts:

https://github.com/user-attachments/assets/abf97a89-9bfa-40f0-a19c-caf7aeb8b4a4

https://github.com/user-attachments/assets/fecb33c2-d31c-4719-a784-6557357aff52

### setup

```
conda create --name bg10 -c conda-forge python=3.10
git clone https://github.com/chernyadev/bigym.git
cd bigym
pip install .

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
sudo apt-get update
sudo apt install xvfb

# training command example
xvfb-run -a python train.py model.hidden_dim=256 model.transformer.enc_layers=3 model.transformer.dec_layers=3 model.transformer.dim_feedforward=1024

# evaluation command example
xvfb-run -a python evaluate.py data.full_demo=false base_dir=exp2 model.use_task_emb=true
```
