import os
import time
import glob
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for, render_template, send_from_directory 

from models import FCN8s, UNet, AUNet, UNetM
from tt import save_load_model
from predict import runPredict

app = Flask(__name__)

msc = { 'unet': UNet(), 'aunet': AUNet(), 'unetm': UNetM(), 'fcn': FCN8s() }
msr = { 'unet': UNet(), 'aunet': AUNet(), 'unetm': UNetM(), 'fcn': FCN8s() }

mns = ['unet', 'aunet', 'unetm', 'fcn']
css = ['crop', 'rescale']
cos = ['gray', 'colourful', 'colourfulmask']

model_dir = './modelbest'
for mn in mns:
	model_path = os.path.join(model_dir, '%s-crop*.pth'%(mn))
	model_paths = glob.glob(model_path)
	if len(model_paths) ==0:
		print('!!!!!! -- no model for %s'%mn)
	else:
		msc[mn] = save_load_model(msc[mn], model_paths[0], "load")

for mn in mns:
	model_path = os.path.join(model_dir, '%s-rescale*.pth'%(mn))
	model_paths = glob.glob(model_path)
	if len(model_paths) ==0:
		print('!!!!!! -- no model for %s'%mn)
	else:
		msr[mn] = save_load_model(msr[mn], model_paths[0], "load")

app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

def allowed_file(filename):
	return '.' in filename and \
			filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/index')
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/submit/', methods=['POST'])
def submit():
	if request.method == 'POST':
		file = request.files['file']
		if file and allowed_file(file.filename):
			file_namee = file.filename.rsplit('.', 1)
			file_name = file_namee[0] + '-%f.'%(time.time()) + file_namee[1]
			file_names = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], secure_filename(file_name))
			if not os.path.exists(os.path.dirname(file_names)):
				os.mkdir(os.path.dirname(file_names))
			file.save(file_names)
			modeln = request.form['model']
			cs = request.form['cs']
			cotype = request.form['cotype']

			pcfg = { 'hwratio': 1.5,
					'save_cs': './static/images/pre_proc.png',
					'save_co': './static/images/pre_out.png',}

			if not (modeln in mns and cs in css and cotype in cos):
				return 'Argument Wrong'
			if runPredict(file_names, msc[modeln] if cs=="crop" else msr[modeln], cs, cotype, pcfg):
				os.remove(file_names)
				return 'success'
			else:
				os.remove(file_names)
				return 'Predict has something wrong!'
		else:
			return 'Failed! Wrong File!'
	else:
		return request.args.__str__()

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8800, debug=True)





