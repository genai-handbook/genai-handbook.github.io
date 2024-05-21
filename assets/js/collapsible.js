document.addEventListener("DOMContentLoaded", function() {
  window.toggleSection = function(header) {
    var content = header.nextElementSibling;
    header.classList.toggle("active");
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  };

  var headers = document.querySelectorAll(".section-header");

  headers.forEach(function(header) {
    header.addEventListener("click", function() {
      toggleSection(this);
    });
  });
});
