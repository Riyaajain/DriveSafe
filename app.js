const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const path = require("path");
const axios = require("axios"); // Make sure axios is installed: npm install axios
const authRoutes = require("./routes/auth");

const app = express();

//  Middleware & Static Setup
app.use(cors());
app.use(express.json());
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.static(path.join(__dirname, "public"))); // Serve HTML, CSS, JS, etc.

//  MongoDB Connection
mongoose.connect("mongodb://localhost:27017/drivesafe", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log("✅ MongoDB connected"))
.catch((err) => console.error("❌ MongoDB connection error:", err));

//  API Routes
app.use("/api", authRoutes);

//  Dashboard Front Page (Landing)
app.get("/", (req, res) => {
  res.render("dashboard/front"); // views/dashboard/front.ejs
});

//  GPS Tracking Page
app.get("/dashboard/tracking", (req, res) => {
  console.log("Navigating to tracking page...");
  res.sendFile(path.join(__dirname, "public", "tracking.html"));
});

//  Drowsiness Detection Page
app.get("/dashboard/drowsiness", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "drowsiness_detector.html"));
});

// Drowsiness Status API Proxy to Flask
app.get("/api/drowsiness-status", async (req, res) => {
  try {
    const response = await axios.get("http://localhost:5000/status"); // Flask endpoint
    res.json(response.data);
  } catch (error) {
    console.error("Error fetching drowsiness status:", error.message);
    res.status(500).json({ error: "Unable to fetch drowsiness status" });
  }

});

// Start the Server
const port = 5000;
app.listen(port, () => {
  console.log(`🌐 Server running at http://localhost:${port}`);
});
