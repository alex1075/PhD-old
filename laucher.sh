#!/bin/bash
echo "Welcome to the program launcher"
echo "Be sure to have the requirements installed"
PS3='Choose what you would like to do today: '
option=("TFTEST" "Benchmark" "Convert dataset to jpeg" "Convert video to jpeg" "Quit")
select fav in "${option[@]}"; do
    case $fav in
        "TFTEST")
            echo "Testing out for compatible GPUs (AMD ROC GPUs won't appear unless Tensorflow compiled to use ROC"
            python Python/TF_test_GPU.py
            ;;
        "Benchmark")
            echo "Benchmarking Tensorflow Setup"
            python Python/benchmark.py
            ;;
        "Convert dataset to jpeg")
            echo "Ensure the dataset path is correct"
            echo "Beginning conversion"
            python Python/convert.py
            exit
            ;;
        "Convert video to jpeg")
            echo "Ensure the video is in the correct place"
            echo "Beginning conversion"
            python Python/make_image.py
            exit
           ;;
        "Quit")
            echo "User requested exit"
            exit
            ;;
        *) echo "invalid option $REPLY";;
    esac
done
