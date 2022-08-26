export function getButtonStyleIfDisabled(file: File | undefined): string {
  if (!file) {
    return "disabled-button";
  }

  return "primary-button";
}

export function isFileUndefined(file: File | undefined): boolean {
  if (!file) {
    return true;
  }
  return false;
}
