@echo off

:: Directly running the easycarla command with specified values
easycarla --host 127.0.0.1 ^
          --port 2000 ^
          --client_timeout 60 ^
          --sync true ^
          --fixed_delta_seconds 0.05 ^
          --fps 20 ^
          --timeout 0.01 ^
          --num_vehicles 30 ^
          --num_pedestrians 40 ^
          --seed 999 ^
          --reset false ^
          --distance 50 ^
          --show_points true ^
          --show_gizmo false ^
          --output_dir data/kitti ^
          --frame_interval 20 ^
          --frame_count 100 ^
          --train_ratio 0.7 ^
          --val_ratio 0.15 ^
          --test_ratio 0.15
