#input_dir=$PWD/$1
input_dir=$1
output_root=$2
stage=${3:-0}
stop_stage=${4:-2}

echo "from ${input_dir} to ${output_root}"
python3 --version

set -euo pipefail

# seperation & segmentation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
echo "Seperating..."
cd seperation
python3 inference_mp.py --filelist_or_dir $output_root/wav --out_dir $output_root --jobs 2 --ckpt_path /data/v-ziqianning/SingingTTS/data_pipeline/ckpts/bs_roformer
cd -
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
echo "Segmenting..."
cd vad
python3 vad_webrtcvad.py --filelist_or_dir ${input_dir}/vocal --out_dir ${output_root}/ --jobs 16
cd -
fi

# ssl
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
echo "Extracting SSL..."
cd ssl
python3 extract_xlsr.py $output_root/vocal_cut $output_root 2    # vocal
python3 extract_xlsr_6l.py $output_root/vocal_cut $output_root 2 # bgm
cd -
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
echo "Quality Metrics..."
cd quality
python3 dnsmos_mp.py --filelist_or_dir $output_root/vocal_cut --text_path $output_root --jobs 8 --ckpt_path /data/v-ziqianning/SingingTTS/data_pipeline/ckpts
python3 pyannote_mp.py --filelist_or_dir $output_root/vocal_cut --text_path $output_root --jobs 8 
cd -
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
echo "Extracting lyrics..."
cd asr
python3 faster_whisper_mp.py --filelist_or_dir $output_root/vocal_cut --text_path $output_root --jobs 2
cd -
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
python3 g2p_en.py $output_root/text.txt $output_root/phoneme.txt
python3 merge_metrics.py --phone $output_root/phoneme.txt --mos $output_root/dnsmos.txt --spk $output_root/spk.txt --output $output_root/data.txt
fi