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

    $('button').on('click', function (e) {
      e.preventDefault();
      console.log('click');
      var avis = $(this).data('avis');
      var photo_id = $('#photo_id').val();
      var pseudo = $('#pseudo').val();
      $.ajax({
        url: "/send",
        method: "POST",
        data: {
          avis: avis,
          photo_id: photo_id,
          pseudo: pseudo
        },
        success: function(data){
          console.log(data);
          if(data != "error"){
            nextImage();
          }else {
            alert("Un pseudo STP")
          }

        }
      });
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
  delete allImage[ran_key];
  return {key:ran_key, url: image};
}
