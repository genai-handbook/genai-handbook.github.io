document.addEventListener("DOMContentLoaded", function() {
  var sidebar = document.getElementById("sidebar");
  var sidebarCollapse = document.getElementById("sidebarCollapse");

  // Toggle sidebar on button click
  sidebarCollapse.addEventListener("click", function() {
    toggleSidebar();
  });

  // Function to toggle the sidebar
  function toggleSidebar() {
    sidebar.classList.toggle("active");
    if (window.innerWidth <= 1024) {
      document.body.classList.toggle("sidebar-open");
    }
  }

  // Expose the toggleSidebar function to the global scope
  window.toggleSidebar = toggleSidebar;
});
