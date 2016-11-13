( function( window ) {
// TO START THE FLASK SERVER, RUN "python /var/www/html/py/flask/flask-test.py"
$('#predict-revenue').click(function(){
 $.ajax({
      type:'GET',
      url:"http://35.161.100.206:8080/flask/"+$('#genre').val()+"/"+$('#actor-one').val()+"/"+$('#budget').val()+"/"+$('#director').val(), // ADD MORE VARIABLES SEPERATED BY "/"
	  async:true,
	  cache: false,
	  timeout:30000,
      success: function(data) {
        swal({
          title: "That movie proabably wont earn:",
          text: "£"+Math.round(data),
          type: "success",
          confirmButtonText: "WOW, let me do another!"
          });
      },
      error: function(request, status, error) {
        console.log(request, status, error);
        alert("Error: " + error + status + request)
      }
   });
});
	
} )( window );
