export CUDA_VISIBLE_DEVICES=1
cd ..

# variables
time=$(date "+%Y%m%d")
exp_name=${time}_"visualize_cmu"
save_dir="runs/"$exp_name

# check save directory
if [ ! -d ${save_dir} ]; then
    mkdir -p ${save_dir}
fi

# scripts
python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/visualize/visualize_cmu.yaml
# nohup python main.py --exp_name ${exp_name} --run_dir ${save_dir} --config configs/visualize/visualize_cmu.yaml > ${save_dir}/out.log 2>&1 &
