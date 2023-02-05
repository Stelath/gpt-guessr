const buttonpressed = document.getElementById("button-click");

buttonpressed.addEventListener("click", () => {
    chrome.tabs.query({active: true, currentWindow: true}, tabs => {
        const url = tabs[0].url;
        let requestText = url;
        console.log(url);
        fetch("http://localhost:8000/", { 
          method: "POST",
        headers: {
          "Content-Type":"application/json"
        },
        body: JSON.stringify({
          request: requestText
        })
      })
      .then(response => response.json())
      .then(data => {
        console.log(data.response);
        buttonpressed.innerHTML = data.response
      });
    });
})
