var allImage = {};

$( document ).ready(function() {
    console.log( "ready!" );

    $.ajax({
      url: "/photo",
      success: function (data) {
        allImage = data;
        console.log(data);
        nextImage();
      }
    });


});

function nextImage(){
  pickRandomImage();
}

function pickRandomImage(){
  var obj_keys = Object.keys(allImage);
  var ran_key = obj_keys[Math.floor(Math.random() *obj_keys.length)];
  image = allImage[ran_key];
  console.log(image);
}
