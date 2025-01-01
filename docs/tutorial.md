### Get Vision Foundation Model Aligned VAE (VA-VAE).

- Download our pre-trained checkpoint from [here]() (to be released) . It is a pre-trained LDM (VQGAN-KL) VAE with 16x downsample ratio and 32-channel latent dimension (f16d32).

- Modify `tokenizer/configs/vavae_f16d32.yaml` to use your own checkpoint path.

- (Optional) Evaluate the reconstruction performance of VA-VAE with the following command. **TODO**

    ```
    bash run_tokenizer_eval.sh tokenizer/configs/vavae_f16d32.yaml
    ```

### Extract ImageNet Latents

- We use VA-VAE to extract latents for all ImageNet images. During extraction, we apply random horizontal flips to maintain consistency with previous works. Run:

- Modify `extract_features.py` to your own data path and {output_path}.
    
    ```
    bash run_extraction.sh tokenizer/configs/vavae_f16d32.yaml
    ```

- (Optional) Also, you can download our pre-extracted ImageNet latents from [here]() (to be released).

### Train LightningDiT

- We provide a feature-rich DiT training and sampling script. For first-time usage, we recommend using the default configurations. We call this optimized configuration ``LightningDiT``.

- However, you still need to modify some necessary paths as required in ``configs/lightningdit_xl_vavae_f16d32.yaml``.

- Run the following command to start training. It train 64 epochs with LightningDiT-XL/1.

    ```
    bash run_train.sh configs/lightningdit_xl_vavae_f16d32.yaml
    ```

- (Optional) Memory Issues: 

    Our training is running with ``bfloat16``. 
    
    We provide ``checkpointing`` functionality. When you encounter GPU memory constraints, please enable it in the config file. While checkpointing theoretically does not affect training results, it may slow down the training speed. 
    
    If checkpointing still doesn't help, we recommend trying smaller model variants. LightningDiT + VA-VAE still shows impressive performance on ``Large`` and ``Base``  scale models. 
    
    Anyway, free feel to train the model that meets your resources and just enjoy the experiments. Hope LightningDiT won't let you down.

### Inference

- Let's see some demo inference results first before we calculate FID score.

    Run the following command:

    ```
    bash run_fast_inference.sh configs/lightningdit_xl_vavae_f16d32.yaml
    ```
    Images will be saved into ``demo_images/demo_samples.png``, e.g. the following one (ONLY 1/21 DiT training epochs!):
    <div align="center">
    <img src="../images/demo_samples_64ep.png" alt="Demo Samples 64ep" width="600">
    </div>

- Calculate FID score:

    ```
    bash run_fid_eval.sh configs/lightningdit_xl_vavae_f16d32.yaml
    ```
    It will provide a reference FID score. For the final reported FID score in the publication, you need to use ADM's evaluation code for standardized testing.