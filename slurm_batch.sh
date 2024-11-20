#!/bin/bash

# 定义参数范围
declare -a delta_mins=(1 2 3)
declare -a delta_maxs=(10 15 20)
declare -a delta_steps=(2 3 4)
declare -a eps_mins=(0.001 0.005 0.01)
declare -a eps_maxs=(0.005 0.01 0.02)

# 输出目录
log_dir="parameter_sweep_logs"
mkdir -p "$log_dir"

# 记录所有提交的作业
echo "Job submissions started at $(date)" > "${log_dir}/submitted_jobs.txt"

# 计数器
job_count=0

# 遍历所有参数组合
for delta_min in "${delta_mins[@]}"; do
    for delta_max in "${delta_maxs[@]}"; do
        # 确保 delta_max > delta_min
        if [ "$delta_max" -le "$delta_min" ]; then
            continue
        fi
        
        for delta_step in "${delta_steps[@]}"; do
            # 确保 delta_step 小于 delta_max - delta_min
            if [ "$delta_step" -ge "$((delta_max - delta_min))" ]; then
                continue
            fi
            
            for eps_min in "${eps_mins[@]}"; do
                for eps_max in "${eps_maxs[@]}"; do
                    # 确保 eps_max < eps_min
                    if (( $(echo "$eps_max >= $eps_min" | bc -l) )); then
                        continue
                    fi
                    
                    # 创建作业名称
                    job_name="adoti_dm${delta_min}_dM${delta_max}_ds${delta_step}_em${eps_min}_eM${eps_max}"
                    
                    # 提交 SLURM 作业
                    sbatch_output=$(sbatch \
                        --job-name="$job_name" \
                        --output="${log_dir}/%x_%j.log" \
                        --error="${log_dir}/%x_%j_err.log" \
                        slurm_job.sh \
                        --mode "AD_OTI" \
                        --regularization "1" \
                        --learning_rate "1" \
                        --adoti_mode "full" \
                        --delta_min "$delta_min" \
                        --delta_max "$delta_max" \
                        --delta_step "$delta_step" \
                        --eps_min "$eps_min" \
                        --eps_max "$eps_max")
                    
                    # 获取作业ID
                    job_id=$(echo "$sbatch_output" | grep -oE '[0-9]+')
                    
                    # 记录提交的作业
                    echo "Submitted job $job_id: $job_name" >> "${log_dir}/submitted_jobs.txt"
                    echo "Parameters: delta_min=$delta_min, delta_max=$delta_max, delta_step=$delta_step, eps_min=$eps_min, eps_max=$eps_max" >> "${log_dir}/submitted_jobs.txt"
                    echo "----------------------------------------" >> "${log_dir}/submitted_jobs.txt"
                    
                    # 增加计数器
                    ((job_count++))
                    
                    # 每提交5个作业等待30秒
                    if [ $((job_count % 5)) -eq 0 ]; then
                        echo "Submitted $job_count jobs, waiting 30 seconds..."
                        sleep 30
                    fi
                done
            done
        done
    done
done

# 打印总结
echo "----------------------------------------" >> "${log_dir}/submitted_jobs.txt"
echo "Total jobs submitted: $job_count" >> "${log_dir}/submitted_jobs.txt"
echo "Job submissions completed at $(date)" >> "${log_dir}/submitted_jobs.txt"

echo "Submitted $job_count jobs in total."
echo "Check ${log_dir}/submitted_jobs.txt for details."