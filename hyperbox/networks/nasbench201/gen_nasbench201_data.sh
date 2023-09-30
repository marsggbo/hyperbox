#!/bin/bash
set -e

pip install peewee
pip install gdown

if [ -z "${NASBENCHMARK_DIR}" ]; then
    NASBENCHMARK_DIR=~/.hyperbox/nasbench201
fi

echo "Downloading NAS-Bench-201..."
if [ -f "nb201.pth" ]; then
    echo "nb201.pth found. Skip download."
else
    gdown https://drive.google.com/uc\?id\=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_ -O nb201.pth
fi

echo "Generating database..."
rm -f ${NASBENCHMARK_DIR}/nasbench201.db ${NASBENCHMARK_DIR}/nasbench201.db-journal
mkdir -p ${NASBENCHMARK_DIR}
python hyperbox/networks/nasbench201/db_gen/db_gen.py nb201.pth
# python hyperbox.networks.nasbench201.db_gen.db_gen nb201.pth
# rm -f nb201.pth