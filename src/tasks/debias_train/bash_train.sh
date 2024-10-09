#!/bin/bash
# for i in {01..10}; 
# do 
#     python main.py --subjects sub-$i; 
# done

list1=$(seq 10 5 50)

list2=$(seq 1 1 10)


for item1 in $list1; do
    for item2 in $list1; do
        for item3 in $list2; do
            for item4 in $list2; do
                python main.py --subjects sub-08  --ksize "$item1,$item2" --sigmaX "$item3,$item4";
            done 
        done
    done
done
