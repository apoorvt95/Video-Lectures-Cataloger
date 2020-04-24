/* =================================================== */
/* ===== Section 1: Require all the dependencies ===== */
/* =================================================== */

const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const upload = multer({ dest: './uploads/'});
const hbs = require('hbs');
const logger = require('morgan');
let fs = require('fs');
const spawn = require("child_process").spawn;
// Define port for app to listen on
const port =  process.env.PORT || 8080;

/* ==================================================== */
/* ===== Section 2: Configure express middlewares ===== */
/* ==================================================== */

const app =  express();
app.use(bodyParser());  // to use bodyParser (for text/number data transfer between clientg and server)
app.set('view engine', 'hbs');  // setting hbs as the view engine
app.use(express.static(__dirname + '/public'));  // making ./public as the static directory
app.set('views', __dirname + '/views');  // making ./views as the views directory
app.use(logger('dev'));  // Creating a logger (using morgan)
app.use(express.json());
app.use(express.urlencoded({ extended: false }));


/* ==================================== */
/* ===== Section 3: Making Routes ===== */
/* ==================================== */

// GET / route for serving index.html file
app.get('/', (req, res) => {
    res.send('Hello World!');
});

// To make the server live
app.listen(port, () => {
    console.log(`App is live on port ${port}`);
});


var mkdirp = require('mkdirp');

var getDirName = require('path').dirname;

var uint8arrayToString = function(data){
    return String.fromCharCode.apply(null, data);
};

// POST /upload for single file upload
/* ===== Make sure that file name matches the name attribute in your html ===== */
app.post('/api', upload.single('myFile'), (req, res) => {

console.log('API hit');

if (req.file) {
        var originalname = req.file.originalname ;
        console.log(req.file);
        fs.renameSync('./uploads/' + req.file.filename, './uploads/' + originalname);
        const pythonProcess = spawn('python3',["/home/ubuntu/MobileComputing/FogServer/script.py", '/home/ubuntu/MobileComputing/FogServer/uploads/' +  originalname]);
        pythonProcess.stdout.on('data', (data) => {
            console.log(String.fromCharCode.apply(null, data));
            res.status(200).json({"tag":String.fromCharCode.apply(null, data)});
        });
	pythonProcess.stderr.on('data', (data) => {
    		// As said before, convert the Uint8Array to a readable string.
    		console.log(uint8arrayToString(data));
	});

//      console.log('Classified tag: Computer Science');
//	res.status(200).json({"convertedText":"Computer Science"});
       
    } else {
        console.log('No File Uploaded');
        var filename = 'FILE NOT UPLOADED';
        var uploadStatus = 'File Upload Failed';
        res.status(202).json({ fileStatus: filename });
    }
 
});

