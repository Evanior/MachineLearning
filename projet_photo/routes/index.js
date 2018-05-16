var express = require('express');
var router = express.Router();

var mongodb = require('mongodb');
var urlMongo = "mongodb://127.0.0.1:27017/";

var mongoClient = mongodb.MongoClient;

var fs = require('fs');
var CsvReadableStream = require('csv-reader');

var inputStream = fs.createReadStream('db_photo_urls.csv', 'utf8');
var images = {};

inputStream
  .pipe(CsvReadableStream({ parseNumbers: true, parseBooleans: true, trim: true }))
  .on('data', function (row) {
    /*mongoClient.connect(urlMongo, function(err, client) {
    	if(err) {
    		console.log("Impossible de se connecter à mongoDB.Erreur:", err);
    	} else {
    		var db = client.db('projet_photo');
    		console.log("connection OK");
        var collection = db.collection('photos');
        var photo = {
    				_id: row[0],
    				url: row[1],
    			};
    			collection.insert(photo, function(err, result) {
    				if(err){
    					console.log(err);
    				} else {
    					console.log(photo, "done");
    				}
    			});
        client.close();
      }
    });*/
    images[row[1]] = row[2];
  })
  .on('end', function (data) {
    console.log('No more rows!');
  });

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Machine Learning'});
});

router.get('/photo', function(req, res, next){
  res.json(images);
});

router.post('/send', function (req, res, next){
  console.log(req.body);
  var pseudo = req.body.pseudo;
  var avis = req.body.avis;
  var photo_id = req.body.photo_id;
  if(pseudo != ""){
    fs.appendFile("data_photo.csv", pseudo+","+photo_id+","+avis+","+new Date().toJSON()+"\n", function(err) {
      if(err) {
        return console.log(err);
      }
      console.log("The file was saved!");
    });
    res.send(req.body);
  }else {
    res.send("error");
  }

});

module.exports = router;
