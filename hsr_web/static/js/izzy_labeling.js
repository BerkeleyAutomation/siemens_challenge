function set_dim(canvas) {
	dim = 420
	canvas.width = dim;
	canvas.height = dim;
}

var imgcanvas = document.getElementById("imgcanvas");
set_dim(imgcanvas);
var imgctx = imgcanvas.getContext("2d");
var bgImage = new Image();
latencyStart = 0;
latencyEnd = 0;
bgImage.onload = function () {
	latencyEnd = performance.now()
	imgctx.drawImage(bgImage, 0, 0, 420, 420);
}

var bboxcanvas = document.getElementById("bboxcanvas");
set_dim(bboxcanvas);
var ctx = bboxcanvas.getContext("2d");

// var confidences = new Image();

// addr = '10.0.0.95'
addr = '0.0.0.0'

workerID = psiTurk.taskdata.get('workerId')
console.log(psiTurk)

//bounding boxes
curr_boxes = [];
drawing = false;
canDraw = true;
tStart = 0;
tEnd = 0;
vector = false;
curr_motion = 0

// labels = ["Wrench", "Hammer", "Screwdriver", "Tape Measure", "Glue", "Tape"];
// labels = ["Screwdriver", "Scrap", "Tube", "Tape"];
labels = ["utility", "bottle", "cup", "fruit", "assemblyPart", "hammer", "scissors", "screwdriver", "tape", "toy", "tube", "wrench"]

var labelHTML = "";
for (i = 0; i < labels.length; i += 1) {
	labelHTML += "<button class='dropmenu-btn' id='labeldrop" + i + "'>" + labels[i] + "</button>\n"
}
document.getElementById("labelmenu").innerHTML = labelHTML;
update_label(labels[0]);

motions = ["Pickup", "Declutter"]
var motionHTML = "";
for (i = 0; i < motions.length; i += 1) {
	motionHTML += "<button class='dropmenu-btn' id='motiondrop" + i + "'>" + motions[i] + "</button>\n"
}
document.getElementById("motionmenu").innerHTML = motionHTML;
update_motion(motions[0]);

hotkeys = ["q", "w", "e", "r", "t"]
colors = ['#FF0000', '#0000FF', '#00FFFF', '#00FF00', '#000000'];
color_ind = 0;

addEventListener("keydown", function (e) {
	for (i = 0; i < hotkeys.length; i++) {
		//32 offset??
		if (e.keyCode == hotkeys[i].charCodeAt(0) - 32) {
			update_label(labels[i]);
		}
	}
	if (e.keyCode == "s".charCodeAt(0) - 32) {
		updateData();
	}
	esc_code = 27
	if (e.keyCode == esc_code) {
		drawing = false;
	}
}, false);

addEventListener("mousedown", function (e) {
	if (canDraw) {
		pos = mouseToPos(e.clientX, e.clientY);
		if (pos) {
			if (drawing) {
				//record the bounding box
				// console.log("drawing")
				x1 = Math.min(old_pose[0], curr_pose[0]);
				x2 = Math.max(old_pose[0], curr_pose[0]);
				y1 = Math.min(old_pose[1], curr_pose[1]);
				y2 = Math.max(old_pose[1], curr_pose[1]);

				addBbox([x1, y1, x2, y2], curr_label, curr_motion);
				drawing = false;
			}

			else if (vector) {
				// console.log("vector")
				x1 = old_pose[0];
				x2 = curr_pose[0];
				y1 = old_pose[1];
				y2 = curr_pose[1];

				addBbox([x1, y1, x2, y2], curr_label, curr_motion);
				vector = false;
			}

			else {
				// console.log(curr_motion == "Pickup")
				if (curr_motion == "Pickup") {
					drawing = true;
				} else {
					vector = true;
				}
				curr_pose = pos;
				old_pose = pos;
			}
		}
	}
}, false);

function addBbox(bounds, label, motion) {
	c = colors[color_ind];
	color_ind = color_ind % colors.length;
	x1 = bounds[0]
	y1 = bounds[1]
	x2 = bounds[2]
	y2 = bounds[3]

	curr_boxes.push([[[x1, y1], [x2, y2]], c, label, motion]);
}

addEventListener("mousemove", function (e) {
	curr_pose = mouseToPos(e.clientX, e.clientY)
}, false);

function clearData() {
	curr_boxes = [];
	// var table = document.getElementById("boxinfo");
	// table.innerHTML = "";
	color_ind = 0;
}

document.getElementById('clear').onclick = function() {
	clearData();
};

document.getElementById('submit').onclick = function() {
	updateData();
};

function update_label(label_val) {
	curr_label = label_val;
	document.getElementById("clabel").innerHTML = curr_label;
}

function update_motion(motion_val) {
	curr_motion = motion_val;
	document.getElementById("mlabel").innerHTML = curr_motion;
}

function fixClosureLabel(val) {
	return function() {
		update_label(val);
	}
}
for (j=0; j<labels.length; j++) {
	document.getElementById('labeldrop' + j).onclick = fixClosureLabel(labels[j]);
}

function fixClosureMotion(val) {
	return function() {
		update_motion(val);
	}
}
for (j=0; j<motions.length; j++) {
	document.getElementById('motiondrop' + j).onclick = fixClosureMotion(motions[j]);
}


var mouseToPos = function(x, y){
	var rect = bboxcanvas.getBoundingClientRect();
	return (x < rect.left || x > rect.right || y > rect.bottom || y < rect.top) ? false : [x - rect.left, y - rect.top];
}

function getRandomInt() {
	max = 65536
	return Math.floor(Math.random() * Math.floor(max));
}

function updateData() {

	tEnd = performance.now()

	feedback = []

	for (i = 0; i < curr_boxes.length; i += 1) {
		datapoint = curr_boxes[i]
		coords = datapoint[0]
		label = curr_label
		motion = curr_motion
		feedback.push({
			key: "coords",
			value: coords
		})
		feedback.push({
			key: "label",
			value: label
		})
		feedback.push({
			key: "wID",
			value: workerID
		})
		feedback.push({
			key: "motion",
			value: motion
		})
	}
	feedback.push({
		key: "milliseconds",
		value: tEnd - tStart
	})
	feedback.push({
		key: "latency",
		value: latencyEnd - latencyStart
	})
	clearData();

	document.getElementById("gif").style.visibility = "visible"
	canDraw = false;

	$.ajax('http://'+addr+':5000/state_feed', {
        type: "GET",
        data: feedback,
		success: function( response ) {
			document.getElementById("gif").style.visibility = "hidden"
			bgImage.src = 'http://' + addr + ':5000/image/' + getRandomInt()
			// confPath = 'http://' + addr + ':5000/confidences/' + getRandomInt()
			// console.log(confidences)
			latencyStart = performance.now()
			canDraw = true;
			tStart = performance.now()
			// update_confidences(confPath)
		}
    });

};

function update_confidences(confPath) {
	document.getElementById("confidences").innerHTML = "<img src=" + confPath + ">"
}

function drawBox(poses)
{
	p1 = poses[0];
	p2 = poses[1];
	ctx.beginPath();
	ctx.moveTo(p1[0], p1[1]);
	ctx.lineTo(p1[0], p2[1]);
	ctx.lineTo(p2[0], p2[1]);
	ctx.lineTo(p2[0], p1[1]);
	ctx.lineTo(p1[0], p1[1]);
	ctx.lineWidth = 3;
	ctx.stroke();

	xh = (p1[0] + p2[0])/2;
	yh = (p1[1] + p2[1])/2;
	ctx.beginPath();
	ctx.moveTo(xh, p1[1]);
	ctx.lineTo(xh, p2[1]);
	ctx.moveTo(p1[0], yh);
	ctx.lineTo(p2[0], yh);
	ctx.lineWidth = 2;
	ctx.stroke();
}

function drawVector(poses)
{
	p1 = poses[0];
	p2 = poses[1];
	ctx.beginPath();
	ctx.moveTo(p1[0], p1[1]);
	ctx.lineTo(p2[0], p2[1]);
	ctx.lineWidth = 3;
	ctx.stroke();

}

// Draw everything
var render = function () {
	// console.log(vector)
	ctx.clearRect(0, 0, bboxcanvas.width, bboxcanvas.height);
	for (var i = 0; i < curr_boxes.length; i++) {
			ctx.strokeStyle =  curr_boxes[i][1];
			if (curr_motion == "Pickup") {
				drawBox(curr_boxes[i][0]);
			} else {
				drawVector(curr_boxes[i][0])
			}
	}
	if (drawing) {
		ctx.strokeStyle = colors[color_ind];
		drawBox([old_pose, curr_pose]);
	}
	if (vector) {
		ctx.strokeStyle = colors[color_ind];
		drawVector([old_pose, curr_pose]);
	}


};

var main = function () {
	requestAnimationFrame(main);
	render();
};

// Cross-browser support for requestAnimationFrame
var w = window;
requestAnimationFrame = w.requestAnimationFrame || w.webkitRequestAnimationFrame || w.msRequestAnimationFrame || w.mozRequestAnimationFrame;
var then = Date.now();
//load labels
function getTextFile(path) {
    var request = new XMLHttpRequest();
    request.open("GET", path, false);
    request.send(null);
    var returnValue = request.responseText;
    return returnValue;
}

//start up
updateData();
main();
