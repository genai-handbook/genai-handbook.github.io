document.addEventListener("DOMContentLoaded", function() {

  var toc = document.getElementById("toc");
  if (!toc) return;

  var headings = document.querySelectorAll("h1, h2, h3, h4, h5, h6");
  var tocContent = "<ul>";
  var currentLevel = 1;
  var chapterNumber = 0;
  var sectionNumbers = [0, 0, 0, 0, 0, 0];

  headings.forEach(function(heading) {
    var level = parseInt(heading.tagName[1]); // Get the heading level (1 for h1, 2 for h2, etc.)

    // Skip numbering for h1 and headings with the "no-numbering" class
    if (level === 1 || heading.classList.contains("no-numbering")) {
      // Close any open <ul> elements before starting a new section
      tocContent += "</ul>".repeat(currentLevel - 1);
      currentLevel = 1;
      
      // Skip TOC generation for headings with "no-toc" class
      if (!heading.classList.contains("no-toc")) {
        tocContent += `<li class="toc-level-${level}"><a href="#${heading.id}">${heading.textContent}</a></li>`;
      }
      
      // Reset section numbers when starting a new section (h1)
      if (level === 1) {
        sectionNumbers.fill(0);
      }
      return;
    }

    // Update chapter numbers for h2 and beyond
    if (level === 2) {
      chapterNumber++;
      sectionNumbers[1] = chapterNumber;
    } else {
      sectionNumbers[level - 1]++;
    }

    // Reset lower level section numbers
    for (var i = level; i < sectionNumbers.length; i++) {
      sectionNumbers[i] = 0;
    }

    // Generate section number string
    var sectionNumber = 'Chapter ' + sectionNumbers.slice(1, level).filter(num => num > 0).join('.') + ': ';

    // Add section number to the heading text
    var originalText = heading.textContent.trim();
    heading.textContent = sectionNumber + originalText;

    // Set ID for heading if not already set
    if (!heading.id) {
      heading.id = originalText.replace(/\s+/g, '-').toLowerCase();
    }

    // Skip TOC generation for headings with "no-toc" class
    if (heading.classList.contains("no-toc")) {
      return;
    }

    // Generate TOC content
    if (level > currentLevel) {
      tocContent += "<ul>".repeat(level - currentLevel);
    } else if (level < currentLevel) {
      tocContent += "</ul>".repeat(currentLevel - level);
    }

    tocContent += `<li class="toc-level-${level}"><a href="#${heading.id}">${heading.textContent}</a></li>`;
    currentLevel = level;
  });

  // Close any remaining open <ul> elements
  tocContent += "</ul>".repeat(currentLevel - 1);
  toc.innerHTML = tocContent;

  // Handle collapsible sections if needed
  var coll = document.getElementsByClassName("collapsible");
  for (var i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function() {
      this.classList.toggle("active");
      var content = this.nextElementSibling;
      if (content.style.display === "block") {
        content.style.display = "none";
      } else {
        content.style.display = "block";
      }
    });
  }

  var tocLinks = document.querySelectorAll("#toc a");

  tocLinks.forEach(function(link) {
    console.log('Attaching click event to:', link); // Log the links
    link.addEventListener("click", function(event) {
      var targetId = this.getAttribute("href").substring(1);
      var targetElement = document.getElementById(targetId);

      console.log('Clicked link:', this);
      console.log('Target ID:', targetId);
      console.log('Target Element:', targetElement);

      if (targetElement) {
        var section = targetElement.closest('.section');
        if (section) {
          var sectionContent = section.querySelector('.section-content');
          var sectionHeader = section.querySelector('.section-header');

          console.log('Section content:', sectionContent);
          console.log('Section header:', sectionHeader);

          if (sectionContent && sectionContent.style.display !== "block") {
            if (sectionHeader) {
              sectionHeader.classList.add("active");
            }
            sectionContent.style.display = "block";
          }
        }
        var targetPosition = targetElement.getBoundingClientRect().top + window.scrollY;
        window.scrollTo({
          top: targetPosition - 170, // Adjust offset for the header height
          behavior: "smooth"
        });
        console.log('Scrolled to:', targetPosition);
  
        event.preventDefault(); //

        if (window.innerWidth <= 1024) {
          window.toggleSidebar();
        }
      }
    });
  });
});

