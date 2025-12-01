import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
from jinja2 import Environment, FileSystemLoader


@dataclass
class AccuracyReport:
    """
    A class to generate an accuracy report using a Jinja2 template and
    provided data.

    Attributes:
        title (str): The title of the report.
        description (str): A description of the report.
        report_file (str): Path to save the generated report file.
        template_file (str): Name of the template file to use.
        template_dir (Path): Directory containing the template file.
        script_name (str): Name of the script generating the report.
        date (str): The date of the report, defaults to the current date.
        css_file (Path): Path to the CSS file for styling.
    """
    title: str = None
    description: str = None
    report_file: str = None
    template_file: str | Path = None
    template_dir: Path = None
    script_name: str = None
    date: str = field(default_factory=lambda:
                      datetime.datetime.now().strftime("%Y-%m-%d")
                      )
    css_file: str = None

    env: Environment = field(init=False, repr=False, default=None)
    template: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """
        Perform additional initialization after dataclass fields are set.
        Initialize the Jinja2 environment and load the template if parameters
        are provided.
        """
        # Initialize Jinja2 environment and template if template
        # parameters are provided
        if self.template_dir and self.template_file:
            self.template_dir = Path(self.template_dir).resolve()
            # breakpoint()
            self.env = Environment(loader=FileSystemLoader(self.template_dir))
            print(f"self.template_file: {self.template_file}\ndir: {self.template_dir}")
            self.template = self.env.get_template(self.template_file)
            self.css = self.env.get_template(self.css_file)

    def update_attributes(self, **kwargs: Any) -> None:
        """
        Update the attributes of the class from a dictionary of keyword
        arguments.

        Args:
            kwargs (dict): Dictionary of parameters to update on the class.
        """
        for key, value in kwargs.items():
            # Check if the key is a defined attribute
            if key in self.__annotations__:
                if key in ("template_dir", "css_file") and value:
                    value = Path(value).resolve()
                if key == "date" and not value:
                    value = datetime.datetime.now().strftime("%Y-%m-%d")
                setattr(self, key, value)

        # Reinitialize Jinja2 environment and template if updated
        # attributes affect them
        if self.template_dir and self.template_file:
            self.__post_init__()

    def __call__(self, data_dict: Dict[str, Any]) -> str:
        """
        Generate the report by rendering the template with provided data.

        Args:
            data_dict (dict): A dictionary where keys are section titles and
                              values are data frames (or similar) that can be
                              converted to HTML.

        Returns:
            str: Rendered HTML content of the report.
        """
        if not self.template:
            raise ValueError("Template is not initialized. "
                             "Provide 'template_dir' and 'template_file'.")

        # check if custom formula exist:
        if 'custom' in data_dict:
            custom = data_dict.pop('custom')
            custom['table'] = custom['table'].to_html(index=True)
        else:
            custom = None

        # Convert DataFrames or similar objects to HTML
        table_data = {title: df.to_html(index=True)
                      for title, df in data_dict.items()
                      }

        # Read CSS content
        css_content = self.css.render()

        # Prepare data for the template
        description_data = {
            key: getattr(self, key)
            for key in self.__annotations__.keys()
        }
        # Render the template with data
        return self.template.render(
            description_data=description_data,
            table_data=table_data,
            css_content=css_content,
            custom=custom
        )
