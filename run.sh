# Author: Bin Wang
# 05/29/2021

# 1 - carpet 
python ./src/main.py --kernel 7 6 3 2 4 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.2 0.2 0.4 0.5 0.1 --class_names carpet

# 2 - grid
python ./src/main.py --kernel 5 5 5 --num_comp 5 5 5 --layer_of_use 1 2 3 --distance_measure self_ref --hop_weights 0.2 0.2 0.2 --class_names grid

# 3 - leather
python ./src/main.py --kernel 5 5 3 2 2 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.2 0.3 0.7 0.3 0.3 --class_names leather

# 4 - tile
python ./src/main.py --kernel 4 6 6 4 2 --num_comp 3 2 5 5 2 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.3 0.1 0.6 0.4 0.2 --class_names tile

# 5 - wood
python ./src/main.py --kernel 3 7 3 3 2 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.6 0.0 0.0 0.2 0.6 --class_names wood

# 6 - bottle
python ./src/main.py --kernel 7 5 2 2 2 --num_comp 2 3 3 3 3 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.0 0.1 0.0 0.0 0.0 --class_names bottle

# 7 - cable
python ./src/main.py --kernel 3 4 4 2 5 --num_comp 2 2 4 3 5 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.4 0.5 0.2 0.1 0.7 --class_names cable

# 8 - capsule
python ./src/main.py --kernel 4 5 5 --num_comp 4 5 3 --layer_of_use 1 2 3 --distance_measure loc_gaussian --hop_weights 0.0 0.1 0.1 --class_names capsule

# 9 - hazel nut
python ./src/main.py --kernel 3 3 2 4 3 --num_comp 5 2 4 4 3 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.7 0.0 0.7 0.4 0.0 --class_names hazelnut

# 10 - metal nut
python ./src/main.py --kernel 7 3 5 4 2 --num_comp 2 3 4 5 4 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.1 0.7 0.1 0.5 0.2 --class_names metal_nut

# 11 - pill
python ./src/main.py --kernel 3 2 7 2 --num_comp 5 2 3 2 --layer_of_use 1 2 3 4 --distance_measure loc_gaussian --hop_weights 0.6 0.1 0.3 0.3 --class_names pill

# 12 - screw
python ./src/main.py --kernel 2 2 7 5 --num_comp 3 4 4 2 --layer_of_use 1 2 3 4 --distance_measure self_ref --hop_weights 0.7 0.3 0.6 0.3 --class_names screw

# 13 - toothbrush
python ./src/main.py --kernel 3 3 --num_comp 3 3 --layer_of_use 1 2 --distance_measure loc_gaussian --hop_weights 0.1 0.1 --class_names toothbrush

# 14 - transistor
python ./src/main.py --kernel 5 7 3 5 4 --num_comp 4 5 3 5 5 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.4 0.0 0.1 0.2 0.4 --class_names transistor

# 15 - zipper
python ./src/main.py --kernel 5 5 3 2 2 --num_comp 5 5 2 4 4 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.2 0.3 0.7 0.3 0.3 --class_names zipper

