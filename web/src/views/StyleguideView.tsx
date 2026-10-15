import { Button } from "../components/Button";
import { Combobox } from "../components/Combobox";
import { Disclaimer } from "../components/Disclaimer";
import { ProbabilityBar } from "../components/ProbabilityBar";
import { StatRow } from "../components/StatRow";
import { Table } from "../components/Table";

export function StyleguideView() {
  return (
    <section className="view container">
      <h1 className="view__title">Styleguide</h1>
      <p className="view__lede">Component inventory for both themes. Hidden from navigation.</p>

      <h2>Buttons</h2>
      <div className="placeholder-card">
        <Button variant="primary">Primary</Button>{" "}
        <Button variant="secondary">Secondary</Button>{" "}
        <Button variant="text">Text</Button>
      </div>

      <h2>Combobox</h2>
      <Combobox id="styleguide-team" label="Team" options={["Arsenal", "Chelsea", "Liverpool"]} value="Arsenal" onChange={() => undefined} />

      <h2>Probability bar</h2>
      <ProbabilityBar
        segments={[
          { label: "Home", value: 0.542, color: "var(--color-home)" },
          { label: "Draw", value: 0.241, color: "var(--color-draw)" },
          { label: "Away", value: 0.217, color: "var(--color-away)" },
        ]}
      />

      <h2>Stat row</h2>
      <StatRow label="Log loss" value="0.9934" hint="Lower is better." />

      <h2>Table</h2>
      <Table
        caption="Example table"
        columns={[
          { key: "model", header: "Model" },
          { key: "loss", header: "Log loss" },
        ]}
        rows={[
          { model: "LightGBM + Platt", loss: "0.9934" },
          { model: "Bookmaker", loss: "0.9664" },
        ]}
      />

      <h2>Disclaimer</h2>
      <Disclaimer text="Estimates only. Not betting advice." />
    </section>
  );
}
