  // Navigation functionality
  function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
      section.classList.remove('active');
    });

    // Show selected section
    document.getElementById(sectionId).classList.add('active');

    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
      link.classList.remove('active');
    });
    event.target.classList.add('active');

    // Close mobile menu if open
    document.getElementById('navLinks').classList.remove('active');

    // Initialize charts if analytics section is shown
    if (sectionId === 'analytics') {
      setTimeout(initializeCharts, 100);
    }

    // Save the selected section to localStorage
    localStorage.setItem('activeSection', sectionId);
  }

  function toggleMobileMenu() {
    document.getElementById('navLinks').classList.toggle('active');
  }

  // Chart data and initialization
  const chartData = {
    category: {
      labels: ['Entertainment','News & Ploitics','Music','Comedy', 'People & Blogs','Film & Animation', 'Education','Sports','Gaming'],
      datasets: [{ 
        label: 'View Share (%)',
        data: [35, 15, 12, 11, 9, 7, 5, 3],
        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'],
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    time: {
      labels: ['12AM', '1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM', '12PM', '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM'],
      datasets: [{
        label: 'Upload Activity (%)',
        data: [4, 6, 9, 13, 19, 22, 23, 21, 20, 19, 21, 23, 27, 26, 28, 22, 22, 16, 10, 6, 5, 6, 7, 7],
        backgroundColor: 'rgba(255, 99, 132, 0.6)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 3,
        fill: true,
        tension: 0.4
      }]
    },
    days: {
      labels: ['0-5 days', '6-10 days', '11-15 days', '16-20 days', '21-25 days', '26-30 days'],
      datasets: [{
        label: 'Videos Trending (%)',
        data: [45, 25, 15, 8, 4, 3],
        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'],
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    weekday: {
      labels: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
      datasets: [{
        label: 'Trending Success Rate (%)',
        data: [91, 88, 85, 95, 100, 97, 82],
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 3,
        fill: true,
        tension: 0.3
      }]
    }
  };

  let chartsInitialized = false;

  function initializeCharts() {
    if (chartsInitialized) return;

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            font: { family: 'Inter', weight: 'bold' }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(0,0,0,0.1)' },
          ticks: { font: { family: 'Inter', weight: '600' } }
        },
        x: {
          grid: { color: 'rgba(0,0,0,0.1)' },
          ticks: { font: { family: 'Inter', weight: '600' } }
        }
      }
    };

    const doughnutOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            font: { family: 'Inter', weight: 'bold' },
            padding: 15
          }
        }
      }
    };

    new Chart(document.getElementById('categoryChart'), {
      type: 'doughnut',
      data: chartData.category,
      options: doughnutOptions
    });

    new Chart(document.getElementById('timeChart'), {
      type: 'line',
      data: chartData.time,
      options: chartOptions
    });

    new Chart(document.getElementById('daysChart'), {
      type: 'doughnut',
      data: chartData.days,
      options: doughnutOptions
    });

    new Chart(document.getElementById('weekdayChart'), {
      type: 'line',
      data: chartData.weekday,
      options: chartOptions
    });

    chartsInitialized = true;
  }

  // Form handling
  const methodUrlRadio = document.getElementById('method_url');
  const methodManualRadio = document.getElementById('method_manual');
  const urlField = document.getElementById('url_field');
  const manualFields = document.getElementById('manual_fields');

  function toggleFields() {
    if (methodUrlRadio.checked) {
      urlField.style.display = 'block';
      manualFields.style.display = 'none';
      document.getElementById('video_url').required = true;
      ['views', 'likes', 'comment_count', 'title', 'category_encoded', 'publish_date'].forEach(id => {
        document.getElementById(id).required = false;
      });
    } else {
      urlField.style.display = 'none';
      manualFields.style.display = 'block';
      document.getElementById('video_url').required = false;
      ['views', 'likes', 'comment_count', 'title', 'category_encoded', 'publish_date'].forEach(id => {
        document.getElementById(id).required = true;
      });
    }
  }
  
  methodUrlRadio.addEventListener('change', toggleFields);
  methodManualRadio.addEventListener('change', toggleFields);
  toggleFields();

  // Close mobile menu when clicking outside
  document.addEventListener('click', function(event) {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.getElementById('navLinks');
    
    if (!navbar.contains(event.target) && navLinks.classList.contains('active')) {
      navLinks.classList.remove('active');
    }
  });

  // On page load, restore the active section from localStorage
  window.addEventListener('load', function() {
    const activeSection = localStorage.getItem('activeSection');
    if (activeSection) {
      showSection(activeSection);
    } else {
      // Default to the first section if none is found
      showSection('home');
    }
  });

