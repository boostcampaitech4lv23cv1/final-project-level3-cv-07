envs="backend frontend cartoonize track"

source activate base

for env in $envs
do
    conda env remove -n $env
done
