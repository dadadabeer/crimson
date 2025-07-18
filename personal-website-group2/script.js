function toggleDarkMode(){
    document.body.classList.toggle("dark-mode");
    const button = document.getElementById("darkModeButton");
    
    if (document.body.classList.contains("dark-mode")){
        button.textContent = "LIGHT MODE";
    }
    else{
        button.textContent = "DARK MODE";
    }
}

let clickCount = 0;

function changeWelcomeMessage(){
    const messages = ["Welcome To My Portfolio!", "Thanks for clicking!", "Youre getting good at this!", "JS is fun, right?", "Keep Exploring" ];
    clickCount = (clickCount+1) % messages.length;
    document.getElementById("welcomeText").textContent = messages[clickCount];
}

let count = 0;
function updateCounter(action){
    if(action === "increase") { 
        count++;
    }
    else if (action === "decrease"){
        count--;
    }
    else{
        count = 0;
    }

    document.getElementById("counterValue").textContent = count;
}

function validateForm(){
    const nameInput = document.getElementById("nameInput").value;
    const validationMessage = document.getElementById("validationMessage");

    if (nameInput.length < 3){
        validationMessage.textContent = "Name must be atleast 3 characters long";
        validationMessage.style.color = "red";
        return false;
    }

    else{
        validationMessage.textContent = "Valid input. Hello " + nameInput + '!';
        validationMessage.style.color = "green";
        return true;
    }
}

function handleContactSubmit(event){
    event.preventDefault();
    const name = document.getElementById("contact-name").value;
    const thankyou = document.getElementById("thank-you");

    thankyou.textContent = `Thanks ${name}, I will be in touch soon!`;
    document.getElementById("contact-form").reset();

}

function generateQuote(){
    const quotes = [
        "Stay curious.",
        "Code is poetry.",
        "The best way to learn is to build.",
        "Keep it simple, silly.",
        "Debugging is learning."
    ];
    const index = Math.floor(Math.random() * quotes.length)
    document.getElementById("quoteDisplay").textContent = quotes[index]
}

function changeBackgroundColor() {
    const colors = ["#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1"];
    const randomColor = Math.floor(Math.random() * colors.length);
    document.body.style.backgroundColor = colors[randomColor];
}

function fetchGreeting(){
    fetch('http://127.0.0.1:5000/hello') //GET REQUEST
        .then(response => response.text()) //parses the text
        .then(data => {
            document.getElementById("greetingResponse").textContent = data; //shows response in html
        })
        .catch(error => console.error("Error", error)); //handles any error
}

function submitToServer(){
    const name = document.getElementById("serverNameInput").value;
    fetch('http://127.0.0.1:5000/submit', {
        method : "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({myname: name})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("serverResponse").textContent = data.message; //i will receive a json respone that will look like {message: hi thanks bye}
    })
    .catch(error => console.error("Error", error)); //handles any error
}


function generateFunFact(){
    fetch('http://127.0.0.1:5000/funfact') //GET REQUEST
        .then(response => response.json()) //parses the text
        .then(data => {
            document.getElementById("funfact").textContent = data.fun_fact; //shows response in html
        })
        .catch(error => console.error("Error", error)); //handles any error
    }

function sendSimplePost(){
    const message = document.getElementById("messageInput").value;
    fetch('http://127.0.0.1:5000/simplepost', {
        method : "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: message})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("messageResponse").textContent = data.response; //i will receive a json respone that will look like {message: hi thanks bye}
    })
    .catch(error => console.error("Error", error)); //handles any error
}

//     '''
//     Task:
//        Create a form with an input field where the user can type a message (like “Hello world!”).
//        When the user clicks the submit button, your code should:
//         Send the message to a Flask POST route.
//        Flask should return a response like “Your message has been received: [message]”.
//        Display the response message on the webpage.
//    '''
   