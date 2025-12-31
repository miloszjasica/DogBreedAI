export function Header() {
  return (
    <div
      style={{
        textAlign: "center",
        marginTop: "25px",
      }}
    >

      <h1
        style={{
          fontSize: "38px",
          fontWeight: "bold",
          background: "linear-gradient(90deg, #3b82f6, #6545b1ff)",
          WebkitBackgroundClip: "text",
          color: "transparent",
          marginBottom: "6px",
        }}
      >
        Dog Breed AI
      </h1>

      <p
        style={{
          color: "#666",
          fontSize: "17px",
        }}
      >
        Upload dog photo, AI predicts the breed
      </p>
    </div>
  );
}
