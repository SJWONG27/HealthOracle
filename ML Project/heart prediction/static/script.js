// Get all pages
const pages = document.querySelectorAll('.page');
// Get the pagination links
const links = document.querySelectorAll('.pagination .page-link');
      
// Show the first page and hide the others
pages[0].style.display = 'block';
for(let i = 1; i < pages.length; i++) {
    pages[i].style.display = 'none';
}

// Add click event listener to the pagination links
links.forEach(link => {
link.addEventListener('click', e => {
e.preventDefault();
// Get the page id from the href attribute
const pageId = e.target.getAttribute('href');
// Hide all pages and show the selected one
pages.forEach(page => page.style.display = 'none');
document.querySelector(pageId).style.display = 'block';
// Add the 'active' class to the selected link
links.forEach(link => link.parentElement.classList.remove('active'));
e.target.parentElement.classList.add('active');
});
});
  
// get the submit button element
const submitBtn = document.getElementById("submitBtn");

// get the input fields by their class
const inputFields = document.getElementsByClassName("form-control");

// add event listeners to the input fields
for (let i = 0; i < inputFields.length; i++) {
  inputFields[i].addEventListener("input", checkInputs);
}

// define the checkInputs function
function checkInputs() {
  let allInputsFilled = true;
  for (let i = 0; i < inputFields.length; i++) {
    if (inputFields[i].value === "") {
      allInputsFilled = false;
      break;
    }
  }
  if (allInputsFilled) {
    // enable the submit button
    submitBtn.disabled = false;
  } else {
    // disable the submit button
    submitBtn.disabled = true;
  }
}

// call the checkInputs function to initialize the state of the submit button
checkInputs();
