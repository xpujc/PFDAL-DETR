import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('')
    # model.load('') # loading pretrain weights
    model.train(data=r'',
                cache=False,
                imgsz=640,
                epochs=250,
                batch=4,
                workers=0,
                device='',
                # resume='', # last.pt path
                project='',
                name='',
                )