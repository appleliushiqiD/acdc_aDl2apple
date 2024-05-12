window.addEventListener("load", () => {

	var simageId = document.getElementById("input_show");
	var testSegId = document.getElementById("testSegImg");
	var formId = document.getElementById("testform");

	var sfileId = document.getElementById("input_file");

	formId.onreset = function () {
		simageId.src = "";
		testSegId.src = "";
	};

	sfileId.onchange = function () {
		testSegId.src = "";
		var file = this.files;
		var freader = new FileReader();
		freader.readAsDataURL(file[0]);
		freader.onload = function () {
			simageId.src = freader.result;
		};
	};

	function sendData() {
		const XHR = new XMLHttpRequest();
		const FD = new FormData(formId);

		XHR.addEventListener("load", (event) => {
			if (event.target.responseText == 'success') {
				// alert('Success!');

				// var simageId = document.getElementById("input_show");
				// var testSegId = document.getElementById("testSegImg");

				var ts = new Date().getTime();

				simageId.src = "./static/images/pre_proc.png?t=" + ts;
				testSegId.src = "./static/images/pre_out.png?t=" + ts;
			} else { alert(event.target.responseText); }
			// alert("Sent data!");
		});

		XHR.addEventListener("error", (event) => {
			alert("Something Wrong for Request! ");
		});

		XHR.open("POST", "http://127.0.0.1:8800/submit");

		XHR.send(FD);

	}
	
	formId.addEventListener("submit", (event)=> {

		event.preventDefault();
		//alert('submit');
		sendData();

		// var sfileId = document.getElementById("input_file");
		sfileId.value = "";
	});
	
});

window.addEventListener("beforeunload", (event) => {
	
	var sfileId = document.getElementById("input_file");

	sfileId.value = "";
	simageId.src = "";
	testSegId.src = "";
	// Chrome requires returnValue to be set.
	event.returnValue = "";
});


