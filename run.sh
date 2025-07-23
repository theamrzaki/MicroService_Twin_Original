#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='Normal-Recreation'
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='false' --req_loss_approach='Normal-Recreation'
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='false' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='false' --req_loss_approach='FreDF-style' --rec_lambda=0.5 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='false' --req_loss_approach='FreDF-style' --rec_lambda=0.1 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='False' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=0.5
#python3 main.py --FREQ_DOMAIN='encoder_decoder'

#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=0.1
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=0.5 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=0.1 --auxi_lambda=1.0
#python3 main.py --FREQ_DOMAIN='FITS' --MULTI_FITS='true' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=0.5

#python3 main.py --FREQ_DOMAIN='FITS_LPF' --MULTI_FITS='true' --req_loss_approach='Normal-Recreation' --rec_lambda=1.0 --auxi_lambda=0.5
#python3 main.py --FREQ_DOMAIN='FITS_Pai' --MULTI_FITS='true' --req_loss_approach='Normal-Recreation' --rec_lambda=1.0 --auxi_lambda=0.5


#python3 main.py --FREQ_DOMAIN='iTransformer' --MULTI_FITS='false' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=0.1
#python3 main.py --FREQ_DOMAIN='DLinear' --MULTI_FITS='false' --req_loss_approach='FreDF-style' --rec_lambda=1.0 --auxi_lambda=0.1

#python3 main.py --FREQ_DOMAIN='FITS' --fewshot_ratio=0.3
#python3 main.py --FREQ_DOMAIN='encoder_decoder' --fewshot_ratio=0.3
#python3 main.py --FREQ_DOMAIN='FITS' --fewshot_ratio=0.1
#python3 main.py --FREQ_DOMAIN='encoder_decoder' --fewshot_ratio=0.1

#python3 main.py --FREQ_DOMAIN='FITS' --rec_lambda=1.0 --auxi_lambda=0.5
#python3 main.py --FREQ_DOMAIN='TimesNet' --rec_lambda=1.0 --auxi_lambda=0.1
#python3 main.py --FREQ_DOMAIN='FreTS' --rec_lambda=1.0 --auxi_lambda=0.1

#python3 main.py --FREQ_DOMAIN='FEDformerModel' --rec_lambda=1.0 --auxi_lambda=0.1

#python3 main.py --FREQ_DOMAIN='FITS_chebyshev' --req_loss_approach='chebyshev-style' --rec_lambda=1.0 --auxi_lambda=0.1
#python3 main.py --FREQ_DOMAIN='FITS_lag' --req_loss_approach='lag-style' --rec_lambda=1.0 --auxi_lambda=0.1
#python3 main.py --FREQ_DOMAIN='FITS_hermite' --req_loss_approach='hermite-style' --rec_lambda=1.0 --auxi_lambda=0.1


#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.3 --random_seed=43
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.3 --random_seed=44
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.3 --random_seed=45
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.3 --random_seed=46
#
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.5 --random_seed=43
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.5 --random_seed=44
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.5 --random_seed=45
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.5 --random_seed=46
#
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.7 --random_seed=43
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.7 --random_seed=44
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.7 --random_seed=45
#python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=0.7 --random_seed=46

python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=1.0 --random_seed=43
python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=1.0 --random_seed=44
python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=1.0 --random_seed=45
python3 main.py --FREQ_DOMAIN='FITS_Legendre' --req_loss_approach='Legendre-style' --fewshot_ratio=1.0 --random_seed=46

python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.3 --random_seed=43
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.3 --random_seed=44
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.3 --random_seed=45
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.3 --random_seed=46

python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.5 --random_seed=43
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.5 --random_seed=44
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.5 --random_seed=45
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.5 --random_seed=46

python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.7 --random_seed=43
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.7 --random_seed=44
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.7 --random_seed=45
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=0.7 --random_seed=46

python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=1.0 --random_seed=43
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=1.0 --random_seed=44
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=1.0 --random_seed=45
python3 main.py --FREQ_DOMAIN='FITS' --req_loss_approach='FreDF-style' --fewshot_ratio=1.0 --random_seed=46


