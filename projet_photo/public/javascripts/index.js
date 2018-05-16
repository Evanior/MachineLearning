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
  $('.carousel-item').removeClass('active');
  image = pickRandomImage();
  console.log(image);
  url = image.url;
  key = image.key;
  $('.carousel-item img').attr('src', url);
  $('#photo_id').val(image.key);
  $('.carousel-item').addClass('active');
}

function pickRandomImage(){
  var obj_keys = Object.keys(allImage);
  var ran_key = obj_keys[Math.floor(Math.random() *obj_keys.length)];
  image = allImage[ran_key];
  return {key:ran_key, url: image};
}
