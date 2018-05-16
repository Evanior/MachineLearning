var express = require('express');
var router = express.Router();

var mongodb = require('mongodb');
var urlMongo = "mongodb://127.0.0.1:27017/";

var mongoClient = mongodb.MongoClient;

var fs = require('fs');
var CsvReadableStream = require('csv-reader');

var inputStream = fs.createReadStream('db_photo_urls.csv', 'utf8');
var images = {images:[]};

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
    images.images[row[0]] = row[1];
  })
  .on('end', function (data) {
    console.log('No more rows!');
  });

/* GET home page. */
router.get('/', function(req, res, next) {
  //console.log(images);
  res.render('index', { title: 'Index', images: images });
});

router.get('/photo', function(req, res, next){
  console.log(images);
  res.json(images);
});

module.exports = router;
