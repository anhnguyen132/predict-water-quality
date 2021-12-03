#!/bin/sh
DATASET_PATH=$1

for cname in AL BD CD CM GH GU IA KE KM LB LS MD MM MW MZ NG PE PH SN TG TJ UG ZM ZW BF TZ ML ET
do
    mv $DATASET_PATH/$cname $DATASET_PATH/train
done

for cname in BJ BO CO DR GA GN GY HT NM SL TD
do
    mv $DATASET_PATH/$cname $DATASET_PATH/val
done

for cname in AM AO BU CI EG KY NP PK RW SZ
do 
    mv $DATASET_PATH/$cname $DATASET_PATH/test
done