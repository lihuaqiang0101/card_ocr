from flask import Flask, request,jsonify
import paddlehub as hub
import cv2
import numpy as np
import os
from OCR import getCards
from OCR import getText

ocr = hub.Module(name="chinese_ocr_db_crnn_server")#chinese_ocr_db_crnn_mobile server
app = Flask(__name__)

count = 0
@app.route("/", methods=["POST"])
def read_asset():
    global count
    try:
        image = request.data
        image = cv2.imdecode(np.asarray(bytearray(image), dtype='uint8'), cv2.IMREAD_COLOR)
        cv2.imwrite(r'C:\xingshizheng\images\{}.jpg'.format(count),image)
        name, Cards = getCards(r'C:\xingshizheng\images\{}.jpg'.format(count))
        Cards = getText(name, Cards)
        np_images = [Cards]
        results = ocr.recognize_text(
            images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
            use_gpu=True,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
            box_thresh=0.5,  # 检测文本框置信度的阈值；
            text_thresh=0.2)  # 识别中文文本置信度的阈值；
        for result in results:
            data = result['data']
            cphm = data[0]['text']
            cllx = data[1]['text']
            syr = data[2]['text']
            zz = data[3]['text']
            syxz = data[4]['text']
            ppxh = data[5]['text']
            clsbdh = data[6]['text']
            fdjhm = data[7]['text']
            zcrq = data[8]['text']
            fzrq = data[9]['text']
            Dic = {'车牌号码': cphm, '车辆类型': cllx, '所有人': syr, '住址': zz, '使用性质': syxz,
                   '品牌型号': ppxh, '车辆识别代号': clsbdh, '发动机号码': fdjhm, '注册日期': zcrq, '发证日期': fzrq}
        count += 1
        return jsonify(Dic)
    except:
        result = jsonify({'code': 500, 'result':'not a driving license'})
        return result


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    app.run(host='192.168.8.101',port='5002')