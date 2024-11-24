document.addEventListener("DOMContentLoaded", function () {
    const searchButton = document.getElementById("search-button");
    const searchInput = document.getElementById("search-input");
    const waterLevel = document.getElementById("water-level");
    const mapImg = document.getElementById("map-img");
    const historicalDataImg = document.getElementById("historical-data-img");
    const heatMapsImg = document.getElementById("heat-maps-img");
  
    const fetchImages = (query) => {
      const imageMap = {
        "12345": {
          map: "assets/map-12345.png",
          historical: "assets/historical-12345.png",
          heatmap: "assets/heatmap-12345.png",
        },
        "New York": {
          map: "assets/map-ny.png",
          historical: "assets/historical-ny.png",
          heatmap: "assets/heatmap-ny.png",
        },
      };
  
      return imageMap[query] || null;
    };
  
    searchButton.addEventListener("click", () => {
      const query = searchInput.value.trim();
  
      if (!query) {
        alert("Please enter a valid ZIP code or city name.");
        return;
      }
  
      const images = fetchImages(query);
  
      if (images) {
        waterLevel.textContent = "Water Level: Elevated";
        mapImg.src = images.map;
        historicalDataImg.src = images.historical;
        heatMapsImg.src = images.heatmap;
      } else {
        alert("No data available for the entered ZIP code or city name.");
      }
    });
  });