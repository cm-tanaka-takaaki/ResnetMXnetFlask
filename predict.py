import numpy as np
import cv2, os
import mxnet as mx
import argparse
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

@app.route("/")
def index():
    return render_template("upload.html")

@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    synset = [l.strip() for l in open('synset.txt').readlines()]
    img = cv2.cvtColor(cv2.imread(destination), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (c, h,w) order
    img = img[np.newaxis, :]  # extend to (n, c, h, w)

    ctx = mx.cpu()
    epoch = 0
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', epoch)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    arg_params["data"] = mx.nd.array(img, ctx)
    arg_params["softmax_label"] = mx.nd.empty((1,), ctx)
    exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
    exe.forward(is_train=False)

    prob = np.squeeze(exe.outputs[0].asnumpy())
    sorted_porb = np.sort(prob)[::-1]
    pred = np.argsort(prob)[::-1]
    top_list =['Category: ' + str(synset[pred[i]]) + ' | Probability: ' + str(sorted_porb[i])  for i in range(5)]
    top1 = top_list[0]
    top2 = top_list[1]
    top3 = top_list[2]
    top4 = top_list[3]
    top5 = top_list[4]
    return render_template("complete.html", image_name=filename, top1=top1, top2=top2, top3=top3, top4=top4, top5=top5)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4555, debug=True)

