## Setup instructions for your own SageMaker Notebook instances

The following instructions can help you set up your own [SageMaker Notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) from the tutorial.

#### In Amazon SageMaker console, create a lifetime configuration named `d2lnlp`

For `Create notebook` script, enter the following

```bash
#!/bin/bash
sudo -u ec2-user -i <<'EOF'

env_name=conda_mxnet_p36
env_name_short=mxnet_p36

# Create a new environment
source /home/ec2-user/anaconda3/bin/activate $env_name_short

# Install mxnet and d2l
pip uninstall -y mxnet-cu100mkl
pip install d2l==0.10.1
pip install mxnet-cu100==1.5.0
pip install gluonnlp==0.7.1
pip install nltk
pip install sacremoses

# Get notebooks from git
cd ~/SageMaker
rm -rf *
git clone https://github.com/astonzhang/KDD19-tutorial.git

# Set notebooks' default kernel to $env_name
cd KDD19-tutorial
for f in 0*/*ipynb; do
    sed -i s/\"language_info\":\ {/\"kernelspec\":\ {\"display_name\":\ \"$env_name\",\"language\":\ \"python\",\ \"name\":\ \"$env_name\"\},\"language_info\":\ {/g $f
done

source /home/ec2-user/anaconda3/bin/deactivate

EOF
```

#### Create notebook instance with Lifetime Configuration `d2lnlp`

When you create your own notebook instance, click the `Additional configuration` and specify `d2lnlp` in `Lifetime configuration - optional` section. Launch the notebook instance as usual.
