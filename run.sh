#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='Normal-Recreation'
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='false' --req_loss_approach='Normal-Recreation'
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='false' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='false' --req_loss_approach='FreDF-style' --rec_lambda=0.5 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='false' --req_loss_approach='FreDF-style' --rec_lambda=0.1 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='False' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=0.5
#python3 main.py --FREQ_DOMAIN='encoder_decoder'

python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=0.1
python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=1.0
python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=0.5 --auxi_lambda=1.0
python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=0.1 --auxi_lambda=1.0
python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=0.5
