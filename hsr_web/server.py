import sys

#sys.path.append('/Users/michaelluskey/Documents/RL/LFD/lapmaster1.1/')
#sys.path.append('/home/laskeymd/RL')


import IPython
import cv2
import numpy as np
from PIL import Image
from flask import send_file
import os

from flask import Blueprint, render_template, request, jsonify, Response, abort, current_app,make_response
from jinja2 import TemplateNotFound
from flask import Flask
#from flask.ext.jsonpify import jsonify
from functools import wraps,update_wrapper
from sqlalchemy import or_

from psiturk.psiturk_config import PsiturkConfig
from psiturk.experiment_errors import ExperimentError
from psiturk.user_utils import PsiTurkAuthorization, nocache
from datetime import timedelta

# # Database setup
from psiturk.db import db_session, init_db
from psiturk.models import Participant
from json import dumps, loads
from cStringIO import StringIO
import cv2
# load the configuration options
config = PsiturkConfig()
config.load_config()
myauth = PsiTurkAuthorization(config)  # if you want to add a password protect route use this
#api
import Pyro4

#
# explore the Blueprint
custom_code = Blueprint('custom_code', __name__, template_folder='templates', static_folder='static')
custom_code = Flask(__name__)
custom_code.config['PROPAGATE_EXCEPTIONS'] = True

#rc11 = pickle.load(open('/Users/michaelluskey/Documents/RL/LFD/AMT_Experiment/RoboCont.p'))

def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

sharer = Pyro4.Proxy("PYRONAME:shared.server")

def gen_image():
    global sharer
    name = sharer.get_img()

    if (name == None):
        name = "/home/ajaytanwani/PycharmProjects/siemens_challenge/web_default.png"
    print "IMAGE NAME ", name
    frame = open(name, "rb").read()

    return (b'--frame\r\n'
           b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

@custom_code.route('/image/<string:id>')
@crossdomain(origin='*')
def image_get(id):
    return Response(gen_image(),mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_confidences():
#     global sharer
#     name = sharer.get_confidences()
#
#     if (name == None):
#         name = "/home/autolab/Workspaces/michael_working/IL_ROS_HSR/shared_data/img.png"
#     print "IMAGE NAME ", name
#     frame = open(name, "rb").read()
#
#     return (b'--frame\r\n'
#            b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
#
# @custom_code.route('/confidences/<string:id>')
# @crossdomain(origin='*')
# def conf_get(id):
#     return Response(gen_confidences(),mimetype='multipart/x-mixed-replace; boundary=frame')

# labelclasses = ["Wrench", "Hammer", "Screwdriver", "Tape Measure", "Glue", "Tape"] #preserve js ordering
#labelclasses = ["Screwdriver", "Scrap", "Tube", "Tape"]
labelclasses = ["utility", "bottle", "cup", "fruit", "assemblyPart", "hammer", "scissors", "screwdriver", "tape", "toy", "tube", "wrench"]
motions = ["Pickup", "Declutter"]

@custom_code.route('/state_feed')
@crossdomain(origin='*')
def state_feed():
    global sharer

    if len(request.args) != 0:
        data = dict(request.args)['undefined']
        latency = data.pop()
        milliseconds = data.pop()

        print("DATA BELOW")

        if len(data) != 0:
            objects = []
            #group by 3s
            for datapoint in zip(*[data[i::4] for i in range(4)]):
                obj = {}
                num_str = datapoint[0]
                num_int = [int(float(el)) for el in num_str.split(',')]

                print datapoint

                obj['coords'] = num_int
                obj['class'] = labelclasses.index(datapoint[1])
                obj['wID'] = datapoint[2]
                obj['motion'] = motions.index(datapoint[3])
                objects.append(obj)

            label_data = {}
            label_data['num_labels'] = len(objects)
            label_data['objects'] = objects
            label_data['time'] = milliseconds
            label_data['latency'] = latency

            sharer.set_label_data(label_data)
            sharer.set_labeled(True)

    print("obj properties: " + sharer.display())

    print("server waiting")
    while not sharer.is_img_ready():
        pass

    sharer.set_img_ready(False)

    return jsonify(result={"status": 200})

if __name__ == '__main__':
    print "running"

    custom_code.run(host='0.0.0.0', threaded = True)
