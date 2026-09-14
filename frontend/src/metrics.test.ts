import { describe, expect, it } from "vitest";
import { improvementLabel } from "./metrics";

describe("improvementLabel", () => {
  it("does not hide regression behind a positive-looking score", () => {
    expect(improvementLabel(-12.34)).toBe("12.3% worse");
  });
});
