<!DOCTYPE HTML>
<html>
<head>
 <meta charset="utf-8">
 <meta http-equiv="X-UA-Compatible" content="IE=edge">
 <title>Real Or Not-Real Prediction</title>
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet"
  integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
 <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"
  integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM"
  crossorigin="anonymous"></script>
 <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
</head>
<body>
 <nav class="navbar navbar-expand-lg navbar-light bg-light">
  <div class="container-fluid">
   <a class="navbar-brand" href="/">Real Or Not-Real Prediction</a>
   <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNavAltMarkup"
    aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
   </button>
   <div class="nav navbar-nav navbar-right" id="navbarNavAltMarkup">
    <div class="navbar-nav">
    </div>
   </div>
  </div>
 </nav>
<br>
 <p style=text-align:center>Real Or Not-Real prediction web application using Machine Learning algorithms. </p>
 <p style=text-align:center>Enter your full tweet and try it.</p>
 <br>
 <div class='container'>
  <form action="/" method="POST">
   <div class="col-three-forth text-center col-md-offset-2">
    <div class="form-group">
     <textarea class="form-control jTextarea mt-3" id="jTextarea'" rows="5" name="text"
      placeholder="Write your text here..." required>{{text}}</textarea>
<br><br>
     <button class="btn btn-primary btn-outline btn-md" type="submit" name="predict">Predict</button>
    </div>
   </div>
  </form>
 </div>
 <br>
 {% if result %}
 <p style="text-align:center"><strong>Prediction : {{result}}</strong></p>
 {% endif %}
<script>
     function growTextarea (i,elem) {
    var elem = $(elem);
    var resizeTextarea = function( elem ) {
        var scrollLeft = window.pageXOffset || (document.documentElement || document.body.parentNode || document.body).scrollLeft;
        var scrollTop  = window.pageYOffset || (document.documentElement || document.body.parentNode || document.body).scrollTop;  
        elem.css('height', 'auto').css('height', elem.prop('scrollHeight') );
          window.scrollTo(scrollLeft, scrollTop);
      };
      elem.on('input', function() {
        resizeTextarea( $(this) );
      });
      resizeTextarea( $(elem) );
  }
  
  $('.jTextarea').each(growTextarea);
</script>
</body>
</html>
