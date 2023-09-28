if [ -z "${NASBENCHMARK_DIR}" ]; then
    NASBENCHMARK_DIR=~/.hyperbox/nasbenchasr/
fi

echo "Downloading NAS-Bench-ASR..."
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-bench-gtx-1080ti-fp32.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-bench-jetson-nano-fp32.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e10-1234.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e40-1234.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e40-1235.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e40-1236.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-e5-1234.pickle
wget https://github.com/SamsungLabs/nb-asr/releases/download/v1.1.0/nb-asr-info.pickle


mkdir -p ${NASBENCHMARK_DIR}
mv nb-asr*.pickle ${NASBENCHMARK_DIR}