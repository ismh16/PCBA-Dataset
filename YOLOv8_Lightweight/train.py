# test
from ultralytics import YOLO

if __name__ == '__main__':
    '''

    # 实验1：yolov8+replknet+scconv
    # Load a model
    model = YOLO('yolov8s-replk-scconv.yaml')  # build a new model from YAML

    # Train the model
    model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_replkscc')
    
    '''

    '''
    # 实验2：yolov8+replknet+ghostconv
    model = YOLO('yolov8s-replk-ghost.yaml')  # build a new model from YAML
    model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_replkghost')
    '''

    '''
    # 实验3：yolov8+replknet+ghostconv
    model = YOLO('yolov8s-scg.yaml')  # build a new model from YAML
    model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_replkghost')
    '''

    '''
    # 实验4：yolov8+replknet+ghostconv
    model = YOLO('yolov8s-rescg.yaml')  # build a new model from YAML
    model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_replkghost')  
    '''

    '''
    # 实验5：yolov8+replknet+ghostconv(p2,head变长) ******好东西
    model = YOLO('yolov8s-replk-ghost-p2.yaml')  # build a new model from YAML
    model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_replkghost_p2')    
    '''

    '''
    # 实验6：yolov8+ssconv+ghostconv(p2,head变长)  ****还是不太行
    model = YOLO('yolov8s-scconv-replk-ghost-p2.yaml')  # build a new model from YAML
    model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_scconvreplkghost_p2')    
    '''

    '''
    # 实验7：实验5在pcba数据集上实验 ******好东西
    model = YOLO('yolov8s-replk-ghost-p2.yaml')  # build a new model from YAML
    model.train(model='', data='mainBoard.yaml', epochs=300, batch=16, imgsz=640, cache='true', workers=4, name='mb_replkghost_p2')    
    '''

    '''
    # 实验8：yolov8+elan+replk+ghostconv(p2,head变长)
    model = YOLO('yolov8s-mobileone-replk-ghost-p2.yaml')  # build a new model from YAML
    model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_mobileonereplkghost_p2')
    '''

    '''
    # 实验9：yolov8+replknet+ghostconv(p2,head变长) + 数据增强
    model = YOLO('yolov8s-replk-ghost-p2.yaml')  # build a new model from YAML
    model.train(model='', data='mainBoard.yaml', epochs=300, batch=16, imgsz=640, cache='true', workers=6, name='mb_replkghost_p2_pro')    
    '''

    # # 实验10：yolov8+ghostconv(p2,head变长)
    # model = YOLO('yolov8s-ghost-p2.yaml')  # build a new model from YAML
    # model.train(model='', data='mainBoard.yaml', epochs=200, batch=8, imgsz=640, cache='true', workers=4,
    #             name='mb_ghost_p2')

    '''
    # 实验11：yolov8(p2,head变长)
    model = YOLO('yolov8s-p2.yaml')  # build a new model from YAML
    model.train(model='', data='mainBoard.yaml', epochs=200, batch=8, imgsz=640, cache='true', workers=5, name='mb_p2')    
    '''

    # # 实验12：yolov8+replk(p2,head变长)
    # model = YOLO('yolov8s-replk-p2.yaml')  # build a new model from YAML
    # model.train(model='', data='mainBoard.yaml', epochs=200, batch=8, imgsz=640, cache='true', workers=5, name='mb_replk_p2')

    # # 实验13：yolov8+ghost(p2,head变长) +数据增强
    # model = YOLO('yolov8s-ghost-p2.yaml')  # build a new model from YAML
    # model.train(model='', data='mainBoard.yaml', epochs=200, batch=8, imgsz=640, cache='true', workers=5, name='mb_ghost_p2_pro')

    # # 实验14：yolov8+replk(p2,head变长) +数据增强
    # model = YOLO('yolov8s-replk-p2.yaml')  # build a new model from YAML
    # model.train(model='', data='mainBoard.yaml', epochs=200, batch=6, imgsz=640, cache='true', workers=5, name='mb_replk_p2_pro')

    # # 实验15：yolov8+ghostconv(p2,head变长)
    # model = YOLO('yolov8s-ghost-p2.yaml')  # build a new model from YAML
    # model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_ghost_p2')

    # # 实验16：yolov8+replknet(p2,head变长)
    # model = YOLO('yolov8s-replk-p2.yaml')  # build a new model from YAML
    # model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_replk_p2')

    # 实验17：yolov8+ghostconv
    model = YOLO('yolov8s-ghost.yaml')  # build a new model from YAML
    model.train(model='', data='VOC.yaml', epochs=300, batch=8, imgsz=640, cache='true', workers=3, name='voc_ghost')


    # # 实验18：yolov8+ghostconv
    # model = YOLO('yolov8s-ghost.yaml')  # build a new model from YAML
    # model.train(model='', data='mainBoard.yaml', epochs=200, batch=6, imgsz=640, cache='true', workers=5,
    #             name='mb_ghost')
