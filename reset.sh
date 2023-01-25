envs="backend frontend cartoonize track"

for env in $envs
do
    conda env remove -n $env
done