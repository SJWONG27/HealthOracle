var startButton = document.querySelector('#logout-button');
startButton.addEventListener('click', function() {
    window.location.href = "logout";
});

var startButton = document.querySelector('#start-button');
startButton.addEventListener('click', function() {
    window.location.href = "heart";
});

var startButton = document.querySelector('#history-button');
startButton.addEventListener('click', function() {
    window.location.href = "history";
});

window.addEventListener('scroll', function() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const windowHeight = window.innerHeight;
    const section1Height = document.getElementById('section-1').offsetHeight;
    const section2Height = document.getElementById('section-2').offsetHeight;
    const section3Height = document.getElementById('section-3').offsetHeight;
    const section4Height = document.getElementById('section-4').offsetHeight;
    const totalHeight = section1Height + section2Height + section3Height + section4Height ;
    const scrollPercentage = scrollTop / (totalHeight - windowHeight);

    if (scrollPercentage < section1Height / totalHeight) {
        document.getElementById('section-1').style.opacity = 1;
        document.getElementById('section-2').style.opacity = 0;
        document.getElementById('section-3').style.opacity = 0;
        document.getElementById('section-4').style.opacity = 0;
    } else if (scrollPercentage < (section1Height + section2Height) / totalHeight) {
        document.getElementById('section-1').style.opacity = 0;
        document.getElementById('section-2').style.opacity = 1;
        document.getElementById('section-3').style.opacity = 0;
        document.getElementById('section-4').style.opacity = 0;
    } else if (scrollPercentage < (section1Height + section2Height + section3Height) / totalHeight) {
        document.getElementById('section-1').style.opacity = 0;
        document.getElementById('section-2').style.opacity = 0;
        document.getElementById('section-3').style.opacity = 1;
        document.getElementById('section-4').style.opacity = 0;
    } else if (scrollPercentage < (section1Height + section2Height + section3Height + section4Height) / totalHeight) {
        document.getElementById('section-1').style.opacity = 0;
        document.getElementById('section-2').style.opacity = 0;
        document.getElementById('section-3').style.opacity = 0;
        document.getElementById('section-4').style.opacity = 1;
    } else {
        document.getElementById('section-1').style.opacity = 0;
        document.getElementById('section-2').style.opacity = 0;
        document.getElementById('section-3').style.opacity = 0;
        document.getElementById('section-4').style.opacity = 1;
    }
});

